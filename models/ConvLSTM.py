from layers.ConvLSTMCell import ConvLSTMCell
import torch    
from torch import nn
from models.model_base import ModelBase


class Model(ModelBase):  
    def __init__(self, configs):
        super(Model, self).__init__()
        
        self.model_name = "ConvLSTM"
        self.input_channels = configs.input_channels
        self.hidden_channels = configs.hidden_channels
        self.kernel_size = configs.kernel_size
        self.num_layers = configs.num_layers
        self.bias = configs.bias
        self.batch_first = configs.batch_first
        self.predict_steps = configs.pred_len
        self.use_multi_heads = configs.use_multi_heads
        self.configs = configs

        # Channel groups info
        self.num_std = getattr(configs, 'num_std', 0)
        self.num_minmax = getattr(configs, 'num_minmax', 0)
        self.num_robust = getattr(configs, 'num_robust', 0)
        self.num_tcc = getattr(configs, 'num_tcc', 0)
        
        self.std_indices = getattr(configs, 'std_cols_indices', [])
        self.minmax_indices = getattr(configs, 'minmax_cols_indices', [])
        self.robust_indices = getattr(configs, 'robust_cols_indices', [])
        self.tcc_indices = getattr(configs, 'tcc_cols_indices', [])
        
        cell_list = []
        for i in range(self.num_layers):
            in_channels = self.input_channels if i == 0 else self.hidden_channels[i - 1]
            cell_list.append(ConvLSTMCell(in_channels, self.hidden_channels[i], self.kernel_size, self.bias))
        
        self.cell_list = nn.ModuleList(cell_list)

        if self.use_multi_heads == 0:
            self.output_conv = nn.Conv2d(
                in_channels=self.hidden_channels[-1],
                out_channels=self.input_channels,
                kernel_size=1,
                padding=0
            )
        else: 
            # 4 Heads
            self.head_std = nn.Conv2d(self.hidden_channels[-1], self.num_std, kernel_size=1, stride=1, padding=0) if self.num_std > 0 else None
            self.head_minmax = nn.Conv2d(self.hidden_channels[-1], self.num_minmax, kernel_size=1, stride=1, padding=0) if self.num_minmax > 0 else None
            self.head_robust = nn.Conv2d(self.hidden_channels[-1], self.num_robust, kernel_size=1, stride=1, padding=0) if self.num_robust > 0 else None
            self.head_tcc = nn.Conv2d(self.hidden_channels[-1], self.num_tcc, kernel_size=1, stride=1, padding=0) if self.num_tcc > 0 else None

    def _apply_heads(self, hidden_state):
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
            
        # Reassemble
        full_output = torch.cat(outputs, dim=1)     
        sort_idx = torch.argsort(torch.tensor(indices)).to(full_output.device)
        return full_output[:, sort_idx, :, :]

    def forward(self, input_tensor, hidden_state=None, mask_true=None, ground_truth=None):
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

        totals_steps = self.configs.his_len + self.configs.pred_len
        predictions = []
        for t in range(totals_steps):
            if self.configs.reverse_scheduled_sampling and ground_truth is not None and mask_true is not None:
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
            
            for i in range(self.num_layers):
                h, c = hidden_state[i]
                h_new, c_new = self.cell_list[i](x, [h, c])
                hidden_state[i] = (h_new, c_new)
                x = h_new
            
            if self.configs.use_multi_heads == 4:
                out = self._apply_heads(h)
            else:
                out = self.output_conv(h)
            if t >= self.configs.his_len:
                predictions.append(out)
        
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
