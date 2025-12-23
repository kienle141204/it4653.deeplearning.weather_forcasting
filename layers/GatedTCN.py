import torch
from torch import nn


class TemporalConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super(TemporalConvLayer, self).__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = (kernel_size - 1) * dilation

        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=self.padding,
        )

    def forward(self, x):
        B, T, N, C = x.shape
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(B * N, T, C)
        x = x.permute(0, 2, 1)

        x = self.conv(x)

        if self.padding > 0:
            x = x[:, :, :-self.padding]

        x = x.permute(0, 2, 1)
        x = x.reshape(B, N, T, -1)
        x = x.permute(0, 2, 1, 3)

        return x


class GatedTCN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super(GatedTCN, self).__init__()
        self.tcn_a = TemporalConvLayer(in_channels, out_channels, kernel_size, dilation)
        self.tcn_b = TemporalConvLayer(in_channels, out_channels, kernel_size, dilation)

    def forward(self, x):
        tanh_out = torch.tanh(self.tcn_a(x))
        sigmoid_out = torch.sigmoid(self.tcn_b(x))
        return tanh_out * sigmoid_out


