import numpy as np
import pandas as pd 
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from utils.timefeatures import time_features
import h5py

class WeatherDataset(Dataset):
    def __init__(self, root_path: str="./data/data.csv", flag="train", size=None, 
                 grid_size=(16, 16), timeenc=0, freq="h", features=True, target="t2m",
                 scaler_std=None, scaler_minmax=None, scaler_robust=None):
        self.root_path = root_path
        self.timeenc = timeenc
        self.freq = freq
        self.flag = flag
        self.features = features
        self.target = target
        if size is None:
            self.his_len = 32
            self.pred_len = 32
        else:
            self.his_len = size[0]
            self.pred_len = size[1]
        self.grid_size = grid_size

        self.scaler_std = scaler_std if scaler_std is not None else StandardScaler()
        self.scaler_minmax = scaler_minmax if scaler_minmax is not None else MinMaxScaler()
        self.scaler_robust = scaler_robust if scaler_robust is not None else RobustScaler()
        
        self.data, self.col_names, self.timestamps = self.__read_data__()
        self.spatial_encoding = self._create_spatial_encoding()
        
        self.temporal_features = self._get_temporal_features(self.timestamps)
    
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
        df["tp"] = df["tp"] * 1000
        
        col_names = [c for c in df.columns if c not in ['valid_time', 'latitude', 'longitude', 'number']]

        if not self.features:
            col_names = self.target
        # print(col_names)
        
        # print("Columns used for model:", col_names)
        std_cols = [col for col in ["t2m", "d2m", "u10", "v10"] if col in col_names]
        minmax_cols = [col for col in ["msl"] if col in col_names]
        robust_cols = [col for col in ["tp"] if col in col_names]
        tcc_col = [col for col in ["tcc"] if col in col_names]
        self.std_cols = std_cols
        self.minmax_cols = minmax_cols
        self.robust_cols = robust_cols
        self.tcc_cols = tcc_col

        
        num_grids = len(df) // (self.grid_size[0] * self.grid_size[1])
        # print(num_grids)
        
        train_size = int(0.7 * num_grids)
        val_size = int(0.15 * num_grids)
        
        train_rows = train_size * self.grid_size[0] * self.grid_size[1]
        val_rows = val_size * self.grid_size[0] * self.grid_size[1]
        
        if self.flag == "train":
            if std_cols:
                self.scaler_std.fit(df[std_cols].iloc[:train_rows])
            if minmax_cols:
                self.scaler_minmax.fit(df[minmax_cols].iloc[:train_rows])
            if robust_cols:
                self.scaler_robust.fit(df[robust_cols].iloc[:train_rows])

        if std_cols and hasattr(self.scaler_std, 'mean_'):  # Check if fitted
            df[std_cols] = self.scaler_std.transform(df[std_cols])
        if minmax_cols and hasattr(self.scaler_minmax, 'data_min_'):  # Check if fitted
            df[minmax_cols] = self.scaler_minmax.transform(df[minmax_cols])
        if robust_cols and hasattr(self.scaler_robust, 'center_'):  # Check if fitted
            df[robust_cols] = self.scaler_robust.transform(df[robust_cols])

        data = df[col_names].values[:num_grids * self.grid_size[0] * self.grid_size[1]].reshape(
            num_grids, self.grid_size[0], self.grid_size[1], -1)
        
        timestamps = timestamps[:num_grids * self.grid_size[0] * self.grid_size[1]].values.reshape(
            num_grids, self.grid_size[0], self.grid_size[1])
        timestamps = timestamps[:, 0, 0]

        train_size = int(0.7 * num_grids)
        val_size = int(0.15 * num_grids)
        
        if self.flag == "train":
            data = data[:train_size]
            timestamps = timestamps[:train_size]
        elif self.flag == "val":
            data = data[train_size:train_size + val_size]
            timestamps = timestamps[train_size:train_size + val_size]
        elif self.flag == "test":
            data = data[train_size + val_size:]
            timestamps = timestamps[train_size + val_size:]
        # print(col_names) 
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
        return len(self.data) - self.his_len - self.pred_len + 1

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.his_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        # Weather data: (T, H, W, C)
        seq_x = torch.FloatTensor(self.data[s_begin:s_end])
        seq_y = torch.FloatTensor(self.data[r_begin:r_end])

        seq_x = seq_x.permute(0, 3, 1, 2)  # (T, C, H, W)
        seq_y = seq_y.permute(0, 3, 1, 2)  # (T, C, H, W)

        seq_x_mark = torch.FloatTensor(self.temporal_features[s_begin:s_end])
        seq_y_mark = torch.FloatTensor(self.temporal_features[r_begin:r_end])
        
        # Spatial encoding: (H, W, D_spatial)
        spatial_enc = self.spatial_encoding

        return index, seq_x, seq_y, seq_x_mark, seq_y_mark, spatial_enc
        
        # return {
        #     'seq_x': seq_x,              # (his_len, H, W, C_weather)
        #     'seq_y': seq_y,              # (pred_len, H, W, C_weather)
        #     'seq_x_mark': seq_x_mark,    # (his_len, D_temporal)
        #     'seq_y_mark': seq_y_mark,    # (pred_len, D_temporal)
        #     'spatial_enc': spatial_enc,  # (H, W, D_spatial)
        # }

    def inverse_transform(self, data):
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
        
        original_shape = data.shape
        data = data.reshape(-1, original_shape[-1]) 
        
        df = pd.DataFrame(data, columns=self.col_names)
        
        std_cols = [col for col in ["t2m", "d2m", "u10", "v10"] if col in df.columns]
        minmax_cols = [col for col in ["msl"] if col in df.columns]
        robust_cols = [col for col in ["tp"] if col in df.columns]
        
        if std_cols and hasattr(self.scaler_std, 'mean_'):  # Check if fitted
            df[std_cols] = self.scaler_std.inverse_transform(df[std_cols])
        if minmax_cols and hasattr(self.scaler_minmax, 'data_min_'):  # Check if fitted
            df[minmax_cols] = self.scaler_minmax.inverse_transform(df[minmax_cols])
        if robust_cols and hasattr(self.scaler_robust, 'center_'):  # Check if fitted
            df[robust_cols] = self.scaler_robust.inverse_transform(df[robust_cols])
        
        # df["t2m"] = df["t2m"] + 273.15
        # df["d2m"] = df["d2m"] + 273.15
        
        result = df.values.reshape(original_shape)
        return result
    
class TrafficDataset(Dataset):
    def __init__(self, root_path: str="./data/BJ13_M32x32_T30_InOut.h5", flag="train",
                 grid_size=(32, 32), size=None, timeenc=0, freq="h",scaler=None):
        self.scaler = StandardScaler()
        self.root_path = root_path
        self.scaler = scaler if scaler is not None else StandardScaler()
        self.flag = flag
        self.grid_size = grid_size

        if size is None:
            self.his_len = 12
            self.pred_len = 12
        else:
            self.his_len = size[0]
            self.pred_len = size[1]
        self.data = self.__read_data__()
    def __read_data__(self):
        file_path = self.root_path

        with h5py.File(file_path, 'r') as f:
            data = np.array(f['data'])

        data = data[:, 0, :, :]
        # print(data.shape)
        num_samples, height, width = data.shape
        train_size = int(num_samples * 0.7)
        val_size = int(num_samples * 0.15)

        train_data = data[:train_size]
        val_data = data[train_size:train_size + val_size]
        test_data = data[train_size + val_size:]

        if self.flag == "train":
            self.scaler.fit(train_data.reshape(-1, 1))
        
        data = self.scaler.transform(data.reshape(-1, 1)).reshape(data.shape)
        if self.flag == "train":
            data = data[:train_size]
        elif self.flag == "val":
            data = data[train_size:train_size + val_size]
        elif self.flag == "test": 
            data = data[train_size + val_size:]

        return data 

    def __len__(self):
        return len(self.data) - self.his_len - self.pred_len + 1

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.his_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = torch.FloatTensor(self.data[s_begin:s_end]).unsqueeze(1)  # (T, 1, H, W)
        seq_y = torch.FloatTensor(self.data[r_begin:r_end]).unsqueeze(1)  # (T, 1, H, W)

        return index, seq_x, seq_y
    
    def inverse_transform(self, data):
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
        
        original_shape = data.shape
        data = data.reshape(-1, 1)
        
        data = self.scaler.inverse_transform(data)
        
        return data.reshape(original_shape)