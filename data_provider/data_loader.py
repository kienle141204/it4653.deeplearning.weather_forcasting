import numpy as np
import pandas as pd 
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

class WeatherDataset(Dataset):
    def __init__(self, root_path: str="./data/data.csv", frag="train", size=None, grid_size=(16, 16)):
        self.root_path = root_path
        self.frag = frag
        if size is None:
            self.seq_len = 32
            self.pred_len = 32
        else:
            self.seq_len = size[0]
            self.pred_len = size[1]
        self.grid_size = grid_size

        self.scaler_std = StandardScaler()  # t2m, d2m, u10, v10
        self.scaler_minmax = MinMaxScaler()  # msl
        self.scaler_robust = RobustScaler()  # tp
        
        self.data, self.col_names = self.__read_data__()
    
    def __read_data__(self):
        df = pd.read_csv(self.root_path)
        df = df.drop(columns=["valid_time", "latitude", "longitude", "number"])
        
        df["t2m"] = df["t2m"] - 273.15  
        df["d2m"] = df["d2m"] - 273.15
        
        col_names = df.columns.tolist()
        
        std_cols = ["t2m", "d2m", "u10", "v10"]
        minmax_cols = ["msl"]
        robust_cols = ["tp"]
        
        df[std_cols] = self.scaler_std.fit_transform(df[std_cols])
        df[minmax_cols] = self.scaler_minmax.fit_transform(df[minmax_cols])
        df[robust_cols] = self.scaler_robust.fit_transform(df[robust_cols])

        num_grids = len(df) // (self.grid_size[0] * self.grid_size[1])
        data = df.values[:num_grids * self.grid_size[0] * self.grid_size[1]].reshape(num_grids, self.grid_size[0], self.grid_size[1], -1)

        train_size = int(0.7 * num_grids)
        val_size = int(0.15 * num_grids)
        
        if self.frag == "train":
            data = data[:train_size]
        elif self.frag == "val":
            data = data[train_size:train_size + val_size]
        elif self.frag == "test":
            data = data[train_size + val_size:]
        
        return data, col_names

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data[s_begin:s_end]  # Shape: (seq_len, grid_size, n_features)
        seq_y = self.data[r_begin:r_end]  # Shape: (pred_len, grid_size, n_features)
        
        seq_x = torch.FloatTensor(seq_x)
        seq_y = torch.FloatTensor(seq_y)
        
        return index, seq_x, seq_y

    def inverse_transform(self, data):
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
        
        original_shape = data.shape
        data = data.reshape(-1, original_shape[-1]) 
        
        df = pd.DataFrame(data, columns=self.col_names)
        
        std_cols = ["t2m", "d2m", "u10", "v10"]
        minmax_cols = ["msl"]
        robust_cols = ["tp"]
        
        df[std_cols] = self.scaler_std.inverse_transform(df[std_cols])
        df[minmax_cols] = self.scaler_minmax.inverse_transform(df[minmax_cols])
        df[robust_cols] = self.scaler_robust.inverse_transform(df[robust_cols])
        
        result = df.values.reshape(original_shape)
        return result