import torch 
from torch import nn

class ModelBase(nn.Module):
    def __init__(self):
        super(ModelBase, self).__init__()
    
    def forward(self, *input):
        raise NotImplementedError("Forward method not implemented.")

    def forcast(self, *input):
        raise NotImplementedError("Forecast method not implemented.")   

    def prepare_data(self, *input):
        raise NotImplementedError("Prepare data method not implemented.")