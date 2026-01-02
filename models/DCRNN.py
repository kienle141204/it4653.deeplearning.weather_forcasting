from layers.DCRNNCell import DCRNNCell
import torch
from torch import nn
from models.model_base import ModelBase
import numpy as np


class Model(ModelBase):
    """
    Diffusion Convolutional Recurrent Neural Network (DCRNN)
    Adapted for grid-based weather forecasting
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        
        self.model_name = "DCRNN"
        self.input_channels = int(configs.input_channels)  # Ensure it's an integer
        # Ensure hidden_channels is a list
        if isinstance(configs.hidden_channels, (list, tuple)):
            self.hidden_channels = [int(h) for h in configs.hidden_channels]  # Ensure all are integers
        else:
            # If single value, create list with that value repeated
            self.hidden_channels = [int(configs.hidden_channels)] * configs.num_layers
        
        # Ensure we have enough hidden_channels for num_layers
        if len(self.hidden_channels) < configs.num_layers:
            # Extend with last value
            last_val = self.hidden_channels[-1] if self.hidden_channels else 64
            self.hidden_channels.extend([last_val] * (configs.num_layers - len(self.hidden_channels)))
        
        self.kernel_size = configs.kernel_size
        self.num_layers = configs.num_layers
        self.bias = configs.bias
        self.batch_first = configs.batch_first
        self.predict_steps = configs.pred_len
        self.use_multi_heads = configs.use_multi_heads
        self.configs = configs
        
        # Number of diffusion steps for spatial information propagation
        self.num_diffusion_steps = getattr(configs, 'num_diffusion_steps', 2)
        
        # Channel groups info
        self.num_std = getattr(configs, 'num_std', 0)
        self.num_minmax = getattr(configs, 'num_minmax', 0)
        self.num_robust = getattr(configs, 'num_robust', 0)
        self.num_tcc = getattr(configs, 'num_tcc', 0)
        
        self.std_indices = getattr(configs, 'std_cols_indices', [])
        self.minmax_indices = getattr(configs, 'minmax_cols_indices', [])
        self.robust_indices = getattr(configs, 'robust_cols_indices', [])
        self.tcc_indices = getattr(configs, 'tcc_cols_indices', [])
        
        # Build DCRNN cells
        cell_list = []
        for i in range(self.num_layers):
            in_channels = self.input_channels if i == 0 else self.hidden_channels[i - 1]
            cell_list.append(
                DCRNNCell(
                    input_channels=in_channels,
                    hidden_channels=self.hidden_channels[i],
                    kernel_size=self.kernel_size,
                    num_diffusion_steps=self.num_diffusion_steps,
                    bias=self.bias
                )
            )
        
        self.cell_list = nn.ModuleList(cell_list)
        
        # Output layer
        if self.use_multi_heads == 0:
            self.output_conv = nn.Conv2d(
                in_channels=self.hidden_channels[-1],
                out_channels=self.input_channels,
                kernel_size=1,
                padding=0
            )
        else:
            # Multi-head output for different normalization types
            self.head_std = nn.Conv2d(
                self.hidden_channels[-1], self.num_std, 
                kernel_size=1, stride=1, padding=0
            ) if self.num_std > 0 else None
            self.head_minmax = nn.Conv2d(
                self.hidden_channels[-1], self.num_minmax, 
                kernel_size=1, stride=1, padding=0
            ) if self.num_minmax > 0 else None
            self.head_robust = nn.Conv2d(
                self.hidden_channels[-1], self.num_robust, 
                kernel_size=1, stride=1, padding=0
            ) if self.num_robust > 0 else None
            self.head_tcc = nn.Conv2d(
                self.hidden_channels[-1], self.num_tcc, 
                kernel_size=1, stride=1, padding=0
            ) if self.num_tcc > 0 else None

    def _apply_heads(self, hidden_state):
        """Apply multi-head output for different normalization types"""
        outputs = []
        indices = []
        
        if self.head_std:
            outputs.append(self.head_std(hidden_state))
            indices.extend(self.std_indices)
        if self.head_minmax:
            outputs.append(self.head_minmax(hidden_state))
            indices.extend(self.minmax_indices)
        if self.head_robust:
            outputs.append(self.head_robust(hidden_state))
            indices.extend(self.robust_indices)
        if self.head_tcc:
            outputs.append(self.head_tcc(hidden_state))
            indices.extend(self.tcc_indices)
        
        # Reassemble in correct order
        full_output = torch.cat(outputs, dim=1)
        sort_idx = torch.argsort(torch.tensor(indices)).to(full_output.device)
        return full_output[:, sort_idx, :, :]

    def forward(self, input_tensor, hidden_state=None, mask_true=None, ground_truth=None):
        """
        Forward pass
        
        Args:
            input_tensor: (B, T, C, H, W) input sequence
            hidden_state: optional initial hidden state
            mask_true: optional mask for scheduled sampling
            ground_truth: optional ground truth for scheduled sampling
        
        Returns:
            predictions: (B, pred_len, C, H, W) predicted sequence
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        # Initialize hidden states
        if hidden_state is not None:
            raise NotImplementedError("Custom hidden state initialization not implemented")
        else:
            hidden_state = self._init_hidden(batch_size=b, image_size=(h, w))

        # Convert mask_true to tensor if it's a numpy array
        if mask_true is not None:
            if isinstance(mask_true, np.ndarray):
                mask_true = torch.from_numpy(mask_true).float()
            # Ensure mask_true is on the same device as input_tensor
            if mask_true.device != input_tensor.device:
                mask_true = mask_true.to(input_tensor.device)
        
        total_steps = self.configs.his_len + self.configs.pred_len
        predictions = []
        
        for t in range(total_steps):
            # Scheduled sampling
            # Support both scheduled_sampling and reverse_scheduled_sampling
            use_sampling = (self.configs.scheduled_sampling or self.configs.reverse_scheduled_sampling)
            if use_sampling and ground_truth is not None and mask_true is not None:
                if t < self.configs.his_len:
                    x = input_tensor[:, t, :, :, :]
                else:
                    mask_true_t = mask_true[:, t - self.configs.his_len, :, :, :]
                    true_x = ground_truth[:, t - self.configs.his_len, :, :, :]
                    x = mask_true_t * true_x + (1 - mask_true_t) * out
            else:
                if t < self.configs.his_len:
                    x = input_tensor[:, t, :, :, :]
                else:
                    x = out
            
            # Process through DCRNN layers
            for i in range(self.num_layers):
                h, c = hidden_state[i]
                h_new, c_new = self.cell_list[i](x, [h, c])
                hidden_state[i] = (h_new, c_new)
                x = h_new
            
            # Generate output
            if self.configs.use_multi_heads == 4:
                out = self._apply_heads(h_new)
            else:
                out = self.output_conv(h_new)
            
            if t >= self.configs.his_len:
                predictions.append(out)
        
        predictions = torch.stack(predictions, dim=1)  # (b, predict_steps, c, h, w)
        
        return predictions

    def _init_hidden(self, batch_size, image_size):
        """Initialize hidden states for all layers"""
        init_states = []
        for i in range(self.num_layers):
            init_states.append(
                self.cell_list[i].init_hidden(batch_size, image_size[0], image_size[1])
            )
        return init_states

