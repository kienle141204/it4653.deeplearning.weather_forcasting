import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_channels, out_channels))
        nn.init.xavier_uniform_(self.weight)
    
    def forward(self, x, adj):
        support = torch.matmul(adj, x)
        return torch.matmul(support, self.weight)


class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, kernel_size),
                            dilation=(1, dilation), padding=(0, (kernel_size-1) * dilation))
    
    def forward(self, x):
        out = self.conv(x)
        return out[:, :, :, :-self.conv.padding[1]]


class GraphWaveNetLayer(nn.Module):
    def __init__(self, in_channels, dilation, kernel_size=2):
        super().__init__()
        self.tcn_filter = TemporalConv(in_channels, in_channels, kernel_size, dilation)
        self.tcn_gate = TemporalConv(in_channels, in_channels, kernel_size, dilation)
        self.gcn = GraphConv(in_channels, in_channels)
        self.residual = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 1))
        self.skip = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 1))

    def forward(self, x, adj):
        f = self.tcn_filter(x)
        g = self.tcn_gate(x)
        tcn_out = torch.tanh(f) * torch.sigmoid(g)
        B, C, N, T = tcn_out.shape
        gcn_input = tcn_out.permute(0, 3, 2, 1).reshape(B*T, N, C)
        gcn_out = self.gcn(gcn_input, adj).reshape(B, T, N, C).permute(0, 3, 2, 1)
        return self.residual(x) + gcn_out, self.skip(gcn_out)

