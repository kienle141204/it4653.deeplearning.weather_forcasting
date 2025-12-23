import torch
from torch import nn

from layers.GatedTCN import GatedTCN
from layers.GCN import GCN


class GraphWaveNetLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super(GraphWaveNetLayer, self).__init__()
        self.gated_tcn = GatedTCN(in_channels, out_channels, kernel_size, dilation)
        self.gcn = GCN(out_channels, out_channels)

    def forward(self, x, adj):
        tcn_out = self.gated_tcn(x)
        gcn_out = self.gcn(tcn_out, adj)
        if x.shape[-1] == gcn_out.shape[-1]:
            return gcn_out + x
        return gcn_out


