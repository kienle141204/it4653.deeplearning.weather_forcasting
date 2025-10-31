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
        self.predict_steps = configs.pred_len
        
        cell_list = []
        for i in range(self.num_layers):
            in_channels = self.input_channels if i == 0 else self.hidden_channels[i - 1]
            cell_list.append(ConvLSTMCell(in_channels, self.hidden_channels[i], self.kernel_size, self.bias))
        
        self.cell_list = nn.ModuleList(cell_list)

        self.output_conv = nn.Conv2d(
            in_channels=self.hidden_channels[-1],
            out_channels=self.input_channels,
            kernel_size=1,
            padding=0
        )

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
            
            hidden_state[layer_idx] = [h, c]

        # decoder
        predictions = []
        
        last_output = self.output_conv(hidden_state[-1][0])
        
        for t in range(self.predict_steps):
            predictions.append(last_output)
            cur_layer_input = last_output
            
            for layer_idx in range(self.num_layers):
                h, c = hidden_state[layer_idx]
                h, c = self.cell_list[layer_idx](
                    input_tensor=cur_layer_input,
                    cur_state=[h, c]
                )
                cur_layer_input = h
                hidden_state[layer_idx] = [h, c]
            
            last_output = self.output_conv(hidden_state[-1][0])
        
        predictions = torch.stack(predictions, dim=1)  # (b, predict_steps, c, h, w)
        
        return predictions

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
