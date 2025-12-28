import torch
import torch.nn as nn
from layers.GraphWaveNetCell import GraphWaveNetLayer
from models.model_base import ModelBase


class Model(ModelBase):
    def __init__(self, configs):
        super(Model, self).__init__()
        
        self.model_name = "GraphWaveNet"
        self.configs = configs
        self.input_channels = configs.input_channels
        # Normalize hidden_channels to match num_layers
        raw_hidden = configs.hidden_channels if isinstance(configs.hidden_channels, list) else [configs.hidden_channels]
        self.num_layers = configs.num_layers
        if len(raw_hidden) < self.num_layers:
            # pad with the last provided value
            raw_hidden = raw_hidden + [raw_hidden[-1]] * (self.num_layers - len(raw_hidden))
        self.hidden_channels = raw_hidden[: self.num_layers]
        self.n_hidden = self.hidden_channels[0] if len(self.hidden_channels) > 0 else 32
        self.kernel_size = configs.kernel_size if hasattr(configs, 'kernel_size') else 2
        self.predict_steps = configs.pred_len
        self.his_len = configs.his_len
        self.num_nodes = configs.grid_size[0] * configs.grid_size[1]
        self.dropout_rate = getattr(configs, 'dropout', 0.3)
        
        # Channel groups info
        self.num_std = getattr(configs, 'num_std', 0)
        self.num_minmax = getattr(configs, 'num_minmax', 0)
        self.num_robust = getattr(configs, 'num_robust', 0)
        self.num_tcc = getattr(configs, 'num_tcc', 0)
        
        self.std_indices = getattr(configs, 'std_cols_indices', [])
        self.minmax_indices = getattr(configs, 'minmax_cols_indices', [])
        self.robust_indices = getattr(configs, 'robust_cols_indices', [])
        self.tcc_indices = getattr(configs, 'tcc_cols_indices', [])
        
        # Initialize adjacency matrix (sẽ được set từ dataset)
        self.register_buffer('adj', torch.eye(self.num_nodes))
        
        # Build architecture từ notebook
        # Input projection
        self.input_proj = nn.Conv2d(self.input_channels, self.n_hidden, kernel_size=(1, 1))
        
        # GraphWaveNet layers với dilation tăng dần
        self.layers = nn.ModuleList([
            GraphWaveNetLayer(self.n_hidden, 2**i, kernel_size=self.kernel_size) 
            for i in range(self.num_layers)
        ])
        
        # Output layer
        if getattr(configs, 'use_multi_heads', 0) == 0:
            self.output = nn.Sequential(
                nn.ReLU(), 
                nn.Conv2d(self.n_hidden, self.n_hidden, kernel_size=(1, 1)),
                nn.ReLU(), 
                nn.Conv2d(self.n_hidden, self.input_channels * self.predict_steps, kernel_size=(1, 1))
            )
        else:
            # Multi-head output
            self.head_std = nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(self.n_hidden, self.n_hidden, kernel_size=(1, 1)),
                nn.ReLU(),
                nn.Conv2d(self.n_hidden, self.num_std * self.predict_steps, kernel_size=(1, 1))
            ) if self.num_std > 0 else None
            self.head_minmax = nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(self.n_hidden, self.n_hidden, kernel_size=(1, 1)),
                nn.ReLU(),
                nn.Conv2d(self.n_hidden, self.num_minmax * self.predict_steps, kernel_size=(1, 1))
            ) if self.num_minmax > 0 else None
            self.head_robust = nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(self.n_hidden, self.n_hidden, kernel_size=(1, 1)),
                nn.ReLU(),
                nn.Conv2d(self.n_hidden, self.num_robust * self.predict_steps, kernel_size=(1, 1))
            ) if self.num_robust > 0 else None
            self.head_tcc = nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(self.n_hidden, self.n_hidden, kernel_size=(1, 1)),
                nn.ReLU(),
                nn.Conv2d(self.n_hidden, self.num_tcc * self.predict_steps, kernel_size=(1, 1))
            ) if self.num_tcc > 0 else None
        
        self.dropout = nn.Dropout(self.dropout_rate)
    
    def set_adjacency(self, adj, device=None):
        if device is None:
            device = next(self.parameters()).device
        
        if not isinstance(adj, torch.Tensor):
            adj = torch.FloatTensor(adj)
        
        adj = adj.to(device)
        d = torch.sum(adj, dim=1)
        d_inv_sqrt = torch.pow(d, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0
        d_mat = torch.diag(d_inv_sqrt)
        self.adj = torch.matmul(torch.matmul(d_mat, adj), d_mat)
    
    def _apply_heads(self, skip_sum):
        """Apply multi-head output từ skip connections"""
        outputs = []
        indices = []
        
        if self.head_std is not None:
            out = self.head_std(skip_sum)  # (batch, num_std * pred_len, nodes, 1)
            b, c, n, t = out.shape
            out = out.squeeze(-1)  # (batch, num_std * pred_len, nodes)
            out = out.view(b, self.num_std, self.predict_steps, n)
            out = out.permute(0, 2, 1, 3)  # (batch, pred_len, num_std, nodes)
            outputs.append(out)
            indices.extend(self.std_indices)
        if self.head_minmax is not None:
            out = self.head_minmax(skip_sum)
            b, c, n, t = out.shape
            out = out.squeeze(-1)
            out = out.view(b, self.num_minmax, self.predict_steps, n)
            out = out.permute(0, 2, 1, 3)
            outputs.append(out)
            indices.extend(self.minmax_indices)
        if self.head_robust is not None:
            out = self.head_robust(skip_sum)
            b, c, n, t = out.shape
            out = out.squeeze(-1)
            out = out.view(b, self.num_robust, self.predict_steps, n)
            out = out.permute(0, 2, 1, 3)
            outputs.append(out)
            indices.extend(self.robust_indices)
        if self.head_tcc is not None:
            out = self.head_tcc(skip_sum)
            b, c, n, t = out.shape
            out = out.squeeze(-1)
            out = out.view(b, self.num_tcc, self.predict_steps, n)
            out = out.permute(0, 2, 1, 3)
            outputs.append(out)
            indices.extend(self.tcc_indices)
        
        # Reassemble in original order
        full_output = torch.cat(outputs, dim=2)  # (batch, pred_len, total_channels, nodes)
        sort_idx = torch.argsort(torch.tensor(indices)).to(full_output.device)
        full_output = full_output[:, :, sort_idx, :]  # (batch, pred_len, input_channels, nodes)
        return full_output
    
    def forward(self, input_tensor, hidden_state=None, mask_true=None, ground_truth=None):
        """
        Forward pass sử dụng architecture từ notebook
        Args:
            input_tensor: (batch, time, channels, height, width) hoặc (batch, time, channels, nodes)
            mask_true: teacher forcing mask (không dùng trong implementation này)
            ground_truth: ground truth for teacher forcing (không dùng trong implementation này)
        """
        # Store original spatial dimensions
        spatial_dims = None
        if input_tensor.dim() == 5:
            # (batch, time, channels, height, width) -> (batch, time, channels, nodes)
            b, t, c, h, w = input_tensor.shape
            spatial_dims = (h, w)
            input_tensor = input_tensor.view(b, t, c, -1)  # (batch, time, channels, nodes)
        
        b, t, c, n = input_tensor.shape
        
        # Lấy chỉ history sequence (his_len)
        x = input_tensor[:, :self.his_len, :, :]  # (batch, his_len, channels, nodes)
        
        # Reshape để phù hợp với notebook format
        # Notebook: (batch, seq_len, nodes, features) -> permute(0, 3, 2, 1) -> (batch, features, nodes, seq_len)
        # Ở đây: (batch, his_len, channels, nodes) -> permute(0, 2, 3, 1) -> (batch, channels, nodes, his_len)
        x = x.permute(0, 2, 3, 1)  # (batch, channels, nodes, his_len)
        
        # Input projection
        x = self.input_proj(x)  # (batch, n_hidden, nodes, his_len)
        
        # Process through GraphWaveNet layers (từ notebook)
        skip_sum = 0
        for layer in self.layers:
            x, skip = layer(x, self.adj)  # x: (batch, n_hidden, nodes, time), skip: same
            skip_sum = skip_sum + skip
            x = self.dropout(x)
        
        # Output projection
        # skip_sum shape: (batch, n_hidden, nodes, time)
        if getattr(self.configs, 'use_multi_heads', 0) != 0:
            # Multi-head output
            predictions = self._apply_heads(skip_sum)  # (batch, pred_len, input_channels, nodes)
        else:
            # Single head output
            # Output shape: (batch, input_channels * pred_len, nodes, time)
            output = self.output(skip_sum)  # (batch, input_channels * pred_len, nodes, time)
            
            # Reshape: (batch, input_channels * pred_len, nodes, time) -> (batch, input_channels, pred_len, nodes, time)
            b_out, c_out, n_out, t_out = output.shape
            output = output.view(b_out, self.input_channels, self.predict_steps, n_out, t_out)
            
            # Mean over temporal dimension (like notebook): (batch, input_channels, pred_len, nodes)
            output = output.mean(dim=-1)
            
            # Permute to (batch, pred_len, input_channels, nodes)
            predictions = output.permute(0, 2, 1, 3)
        
        # Reshape back to spatial if needed
        if spatial_dims is not None:
            predictions = predictions.view(
                b, self.predict_steps, self.input_channels, spatial_dims[0], spatial_dims[1]
            )
        
        return predictions
