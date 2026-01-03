import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DCRNNCell(nn.Module):
    """
    Diffusion Convolutional Recurrent Neural Network Cell
    Adapted for grid-based spatial-temporal data
    """
    def __init__(self, input_channels, hidden_channels, kernel_size, num_diffusion_steps=2, bias=True):
        super(DCRNNCell, self).__init__()
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_diffusion_steps = num_diffusion_steps
        self.padding = kernel_size // 2
        self.bias = bias
        
        # Input-to-hidden transformation
        self.conv_x = nn.Conv2d(
            in_channels=self.input_channels,
            out_channels=4 * self.hidden_channels,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias
        )
        
        # Hidden-to-hidden transformation with diffusion convolution
        # We'll use standard conv2d but can be extended to true graph diffusion
        self.conv_h = nn.Conv2d(
            in_channels=self.hidden_channels,
            out_channels=4 * self.hidden_channels,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias
        )
        
        # Diffusion convolution layers for spatial information propagation
        self.diffusion_convs = nn.ModuleList([
            nn.Conv2d(
                in_channels=self.hidden_channels,
                out_channels=self.hidden_channels,
                kernel_size=self.kernel_size,
                padding=self.padding,
                bias=self.bias
            ) for _ in range(num_diffusion_steps)
        ])
        
    def _apply_diffusion_conv(self, h):
        """
        Apply diffusion convolution to propagate spatial information
        This simulates information diffusion across the grid
        """
        # Multi-step diffusion: each step propagates information to neighbors
        diffused = h
        for conv in self.diffusion_convs:
            diffused = F.relu(conv(diffused)) + diffused  # Residual connection
        
        return diffused
    
    def forward(self, input_tensor, cur_state):
        """
        Forward pass of DCRNN Cell
        
        Args:
            input_tensor: (B, C, H, W) input at current time step
            cur_state: tuple of (h_cur, c_cur) where both are (B, hidden_channels, H, W)
        
        Returns:
            h_next: (B, hidden_channels, H, W) next hidden state
            c_next: (B, hidden_channels, H, W) next cell state
        """
        h_cur, c_cur = cur_state
        
        # Input transformation
        x_conv = self.conv_x(input_tensor)
        x_i, x_f, x_o, x_g = torch.split(x_conv, self.hidden_channels, dim=1)
        
        # Apply diffusion convolution to hidden state
        h_diffused = self._apply_diffusion_conv(h_cur)
        
        # Hidden transformation
        h_conv = self.conv_h(h_diffused)
        h_i, h_f, h_o, h_g = torch.split(h_conv, self.hidden_channels, dim=1)
        
        # LSTM gates
        i_t = torch.sigmoid(x_i + h_i)
        f_t = torch.sigmoid(x_f + h_f)
        o_t = torch.sigmoid(x_o + h_o)
        g_t = torch.tanh(x_g + h_g)
        
        # Update cell and hidden states
        c_next = f_t * c_cur + i_t * g_t
        h_next = o_t * torch.tanh(c_next)
        
        return h_next, c_next
    
    def init_hidden(self, batch_size, height, width):
        """
        Initialize hidden and cell states
        
        Returns:
            tuple of (h_0, c_0) both of shape (B, hidden_channels, H, W)
        """
        device = next(self.parameters()).device
        h_0 = torch.zeros(batch_size, self.hidden_channels, height, width, device=device)
        c_0 = torch.zeros(batch_size, self.hidden_channels, height, width, device=device)
        return (h_0, c_0)

