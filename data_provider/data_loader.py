import numpy as np
import pandas as pd 
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from utils.timefeatures import time_features

class WeatherDataset(Dataset):
    def __init__(self, root_path: str="./data/data.csv", frag="train", size=None, 
                 grid_size=(16, 16), timeenc=0, freq="h"):
        self.root_path = root_path
        self.timeenc = timeenc
        self.freq = freq
        self.frag = frag
        if size is None:
            self.seq_len = 32
            self.pred_len = 32
        else:
            self.seq_len = size[0]
            self.pred_len = size[1]
        self.grid_size = grid_size

        self.scaler_std = StandardScaler()
        self.scaler_minmax = MinMaxScaler()
        self.scaler_robust = RobustScaler()
        
        self.data, self.col_names, self.timestamps = self.__read_data__()
        self.spatial_encoding = self._create_spatial_encoding()
    
    # Fourier Position Encoding for 2D grid
    def _create_spatial_encoding(self):
        y_coords = np.linspace(0, 1, self.grid_size[0])
        x_coords = np.linspace(0, 1, self.grid_size[1])
        
        yy, xx = np.meshgrid(y_coords, x_coords, indexing='ij')
        
        d_model = 8
        pos_enc = np.zeros((self.grid_size[0], self.grid_size[1], d_model))
        
        for i in range(d_model // 4):
            pos_enc[:, :, i*4] = np.sin(yy * (2 * np.pi) * (10 ** (i / (d_model/4))))
            pos_enc[:, :, i*4 + 1] = np.cos(yy * (2 * np.pi) * (10 ** (i / (d_model/4))))
            pos_enc[:, :, i*4 + 2] = np.sin(xx * (2 * np.pi) * (10 ** (i / (d_model/4))))
            pos_enc[:, :, i*4 + 3] = np.cos(xx * (2 * np.pi) * (10 ** (i / (d_model/4))))
        
        return torch.FloatTensor(pos_enc)
    
    def __read_data__(self):
        df = pd.read_csv(self.root_path)
        
        timestamps = pd.to_datetime(df['valid_time'])
        
        cols = list(df.columns)
        cols.remove("valid_time")

        df["t2m"] = df["t2m"] - 273.15  
        df["d2m"] = df["d2m"] - 273.15
        
        col_names = [c for c in df.columns if c not in ['valid_time', 'latitude', 'longitude', 'number']]
        # print("Columns used for model:", col_names)
        
        std_cols = ["t2m", "d2m", "u10", "v10"]
        minmax_cols = ["msl"]
        robust_cols = ["tp"]
        
        df[std_cols] = self.scaler_std.fit_transform(df[std_cols])
        df[minmax_cols] = self.scaler_minmax.fit_transform(df[minmax_cols])
        df[robust_cols] = self.scaler_robust.fit_transform(df[robust_cols])

        num_grids = len(df) // (self.grid_size[0] * self.grid_size[1])
        data = df[col_names].values[:num_grids * self.grid_size[0] * self.grid_size[1]].reshape(
            num_grids, self.grid_size[0], self.grid_size[1], -1)
        
        timestamps = timestamps[:num_grids * self.grid_size[0] * self.grid_size[1]].values.reshape(
            num_grids, self.grid_size[0], self.grid_size[1])
        timestamps = timestamps[:, 0, 0]

        train_size = int(0.7 * num_grids)
        val_size = int(0.15 * num_grids)
        
        if self.frag == "train":
            data = data[:train_size]
            timestamps = timestamps[:train_size]
        elif self.frag == "val":
            data = data[train_size:train_size + val_size]
            timestamps = timestamps[train_size:train_size + val_size]
        elif self.frag == "test":
            data = data[train_size + val_size:]
            timestamps = timestamps[train_size + val_size:]
        
        return data, col_names, timestamps

    def _get_temporal_features(self, timestamps):
        df_stamp = pd.DataFrame({'date': timestamps})
        
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.dt.month
            df_stamp['day'] = df_stamp.date.dt.day
            df_stamp['weekday'] = df_stamp.date.dt.weekday
            df_stamp['hour'] = df_stamp.date.dt.hour
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
        
        return data_stamp

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len


        # Weather data: (T, H, W, C)
        seq_x = torch.FloatTensor(self.data[s_begin:s_end])
        seq_y = torch.FloatTensor(self.data[r_begin:r_end])

        seq_x = seq_x.permute(0, 3, 1, 2)  # (T, C, H, W)
        seq_y = seq_y.permute(0, 3, 1, 2)  # (T, C, H, W)
        
        # Temporal features: (T, D_temporal)
        seq_x_mark = torch.FloatTensor(
            self._get_temporal_features(self.timestamps[s_begin:s_end]))
        seq_y_mark = torch.FloatTensor(
            self._get_temporal_features(self.timestamps[r_begin:r_end]))
        
        # Spatial encoding: (H, W, D_spatial)
        spatial_enc = self.spatial_encoding

        return index, seq_x, seq_y, seq_x_mark, seq_y_mark, spatial_enc
        
        # return {
        #     'seq_x': seq_x,              # (seq_len, H, W, C_weather)
        #     'seq_y': seq_y,              # (pred_len, H, W, C_weather)
        #     'seq_x_mark': seq_x_mark,    # (seq_len, D_temporal)
        #     'seq_y_mark': seq_y_mark,    # (pred_len, D_temporal)
        #     'spatial_enc': spatial_enc,  # (H, W, D_spatial)
        # }

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
        
        # df["t2m"] = df["t2m"] + 273.15
        # df["d2m"] = df["d2m"] + 273.15
        
        result = df.values.reshape(original_shape)
        return result