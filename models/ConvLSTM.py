from layers.ConvLSTMCell import ConvLSTMCell
import torch    
from torch import nn
from models.model_base import ModelBase


class Model(ModelBase):  
    def __init__(self, configs):
        super(Model, self).__init__()
        
        self.input_channels = configs.input_channels
        self.hidden_channels = configs.hidden_channels
        self.kernel_size = configs.kernel_size
        self.num_layers = configs.num_layers
        self.bias = configs.bias
        self.batch_first = configs.batch_first
        
        cell_list = []
        for i in range(self.num_layers):
            in_channels = self.input_channels if i == 0 else self.hidden_channels[i - 1]
            cell_list.append(ConvLSTMCell(in_channels, self.hidden_channels[i], self.kernel_size, self.bias))
        
        self.cell_list = nn.ModuleList(cell_list)

        self.linear = nn.Linear(self.hidden_channels[-1], self.input_channels)

    def forward(self, input_tensor, hidden_state=None):
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        # if not self.return_all_layers:
        #     layer_output_list = layer_output_list[-1:]
        #     last_state_list = last_state_list[-1:]

        last_layer_output = layer_output_list[-1]
        B, T, C, H, W = last_layer_output.shape
        output = self.linear(last_layer_output.reshape(-1, C)).reshape(B, T, -1, H, W)
        return output

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size[0], image_size[1]))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param
    
    # def prepare_data(self, input):
