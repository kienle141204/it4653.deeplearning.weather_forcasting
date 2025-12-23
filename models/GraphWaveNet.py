import torch
from torch import nn

from models.model_base import ModelBase
from layers.GraphWaveNetLayer import GraphWaveNetLayer
from utils.graph_utils import build_adjacency_matrix


class Model(ModelBase):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.model_name = "GraphWaveNet"
        self.input_channels = configs.input_channels
        hidden_channels = getattr(configs, "hidden_channels", 64)
        if isinstance(hidden_channels, list):
            self.hidden_channels = hidden_channels[0] if len(hidden_channels) > 0 else 64
        else:
            self.hidden_channels = hidden_channels
        self.num_layers = getattr(configs, "num_layers", 3)
        self.kernel_size = getattr(configs, "kernel_size", 3)
        self.dropout = getattr(configs, "dropout", 0.3)
        self.predict_steps = configs.pred_len
        self.grid_size = configs.grid_size

        self.adj_method = getattr(configs, "adj_method", "distance")
        self.adj_threshold = getattr(configs, "adj_threshold", 1.5)
        self.adj_k = getattr(configs, "adj_k", 8)

        adj = build_adjacency_matrix(
            self.grid_size,
            method=self.adj_method,
            threshold=self.adj_threshold,
            k=self.adj_k,
        )
        self.register_buffer("adj", torch.FloatTensor(adj))

        self.input_projection = nn.Linear(self.input_channels, self.hidden_channels)

        self.layers = nn.ModuleList()
        dilations = [2 ** i for i in range(self.num_layers)]
        for i in range(self.num_layers):
            self.layers.append(
                GraphWaveNetLayer(
                    in_channels=self.hidden_channels,
                    out_channels=self.hidden_channels,
                    kernel_size=self.kernel_size,
                    dilation=dilations[i],
                )
            )

        self.dropout_layer = nn.Dropout(self.dropout)

        # Improved output layers with deeper architecture for better accuracy
        self.output_norm1 = nn.LayerNorm(self.hidden_channels)
        self.output_fc1 = nn.Linear(self.hidden_channels, self.hidden_channels * 2)
        self.output_norm2 = nn.LayerNorm(self.hidden_channels * 2)
        self.output_fc2 = nn.Linear(self.hidden_channels * 2, self.hidden_channels)
        self.output_norm3 = nn.LayerNorm(self.hidden_channels)
        self.output_dropout = nn.Dropout(self.dropout * 0.5)
        self.output_fc3 = nn.Linear(self.hidden_channels, self.input_channels)

    def forward(self, x, mask_true=None, ground_truth=None):
        B, T, C, H, W = x.shape
        N = H * W

        # Convert input to graph format: (B, T, C, H, W) -> (B, T, N, C)
        x_reshaped = x.permute(0, 1, 3, 4, 2)  # (B, T, H, W, C)
        x_reshaped = x_reshaped.reshape(B, T, N, C)  # (B, T, N, C)

        # Project to hidden space
        x_proj = self.input_projection(x_reshaped)  # (B, T, N, C) -> (B, T, N, hidden_channels)

        # Process historical sequence with residual connections
        skip_connections = [x_proj]
        for i, layer in enumerate(self.layers):
            x_proj = layer(x_proj, self.adj)
            # Add residual connection from input if same shape
            if i == 0:
                skip_connections.append(x_proj)
            else:
                # Add residual from previous layer output
                skip_connections.append(x_proj + skip_connections[-1])

        # Simple sum of skip connections (faster than weighted sum)
        skip_sum = torch.stack(skip_connections, dim=0).sum(dim=0)
        skip_sum = self.dropout_layer(skip_sum)

        # Apply improved deeper output layers
        output = self.output_norm1(skip_sum)
        output = nn.functional.relu(output)
        output = self.output_fc1(output)  # Expand to hidden_channels * 2
        output = self.output_norm2(output)
        output = nn.functional.relu(output)
        output = self.output_fc2(output)  # Back to hidden_channels
        output = self.output_norm3(output)
        output = nn.functional.relu(output)
        output = self.output_dropout(output)
        output = self.output_fc3(output)  # (B, T, N, C)

        # Get last output and convert back to spatial format for autoregressive prediction
        last_output = output[:, -1:, :, :]  # (B, 1, N, C)
        last_output_spatial = last_output.reshape(B, 1, H, W, C).permute(0, 1, 4, 2, 3)  # (B, 1, C, H, W)
        last_output_spatial = last_output_spatial.squeeze(1)  # (B, C, H, W)

        predictions = []
        current_input = last_output_spatial  # (B, C, H, W) - start with last frame in spatial format

        # Convert mask_true to torch if needed
        if mask_true is not None and not isinstance(mask_true, torch.Tensor):
            mask_true = torch.FloatTensor(mask_true).to(x.device)

        for step in range(self.predict_steps):
            # Scheduled sampling: similar to ConvLSTM
            if ground_truth is not None and mask_true is not None:
                # mask_true shape: (B, pred_len, C, H, W)
                mask_step = mask_true[:, step, :, :, :]  # (B, C, H, W)
                true_step = ground_truth[:, step, :, :, :]  # (B, C, H, W)
                # Blend: use ground truth where mask=1, prediction where mask=0
                current_input = mask_step * true_step + (1 - mask_step) * current_input

            # Convert to graph format for processing
            current_input_graph = current_input.permute(0, 2, 3, 1)  # (B, H, W, C)
            current_input_graph = current_input_graph.reshape(B, 1, N, C)  # (B, 1, N, C)

            # Project to hidden space
            current_input_proj = self.input_projection(current_input_graph)  # (B, 1, N, hidden_channels)

            # Process through layers
            layer_output = current_input_proj
            for layer in self.layers:
                layer_output = layer(layer_output, self.adj)

            skip_sum = layer_output + current_input_proj
            skip_sum = self.dropout_layer(skip_sum)
            
            # Apply improved deeper output layers
            pred = self.output_norm1(skip_sum)
            pred = nn.functional.relu(pred)
            pred = self.output_fc1(pred)  # Expand to hidden_channels * 2
            pred = self.output_norm2(pred)
            pred = nn.functional.relu(pred)
            pred = self.output_fc2(pred)  # Back to hidden_channels
            pred = self.output_norm3(pred)
            pred = nn.functional.relu(pred)
            pred = self.output_dropout(pred)
            pred = self.output_fc3(pred)  # (B, 1, N, C)

            # Convert back to spatial format
            pred_spatial = pred.reshape(B, 1, H, W, C).permute(0, 1, 4, 2, 3)  # (B, 1, C, H, W)
            pred_spatial = pred_spatial.squeeze(1)  # (B, C, H, W)

            predictions.append(pred_spatial)
            current_input = pred_spatial  # Use prediction for next step

        # Stack predictions: (B, pred_len, C, H, W)
        predictions = torch.stack(predictions, dim=1)

        return predictions


