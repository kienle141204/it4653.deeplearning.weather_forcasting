import torch 
from torch import nn
from layers.SwinLSTMCell import SwinLSTM

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        img_size = configs.input_img_size
        patch_size = configs.patch_size
        in_chans = configs.input_channels
        embed_dim = configs.embed_dim
        self.depths = configs.depths
        num_heads = configs.heads_number
        window_size = configs.window_size
        drop_rate = configs.drop_rate
        attn_drop_rate = configs.attn_drop_rate
        drop_path_rate = configs.drop_path_rate

        self.predict_steps = configs.pred_len

        self.swin_lstm = SwinLSTM(img_size=img_size, patch_size=patch_size, in_chans=in_chans,
                            embed_dim=embed_dim, depths=self.depths,
                            num_heads=num_heads, window_size=window_size, drop_rate=drop_rate,
                            attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate)


    def forward(self, x):
        B, T, C, H, W = x.shape
        states = [None] * sum(self.depths)     
        outs = []
        for t in range(T):
            frame = x[:, t]                    
            out, states = self.swin_lstm(frame, states)   
            outs.append(out)

        # decoder 
        last_input = x[: ,-1]
        predictions = []
        for t in range(self.predict_steps):
            out, states = self.swin_lstm(last_input, states)
            predictions.append(out)
            last_input = out

        return torch.stack(predictions, dim=1) 
    
