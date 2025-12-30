from data_provider.data_loader import WeatherDataset, TrafficDataset
from torch.utils.data import DataLoader

def data_provider(args, flag="train"):
    if flag == "train":
        train_dataset = WeatherDataset(
            root_path=args.data_path,
            flag="train",
            size=(args.his_len, args.pred_len),
            grid_size=args.grid_size,
            features=args.features,
            target=args.target
        )
        dataset = train_dataset
    else:
        train_dataset = WeatherDataset(
            root_path=args.data_path,
            flag="train",
            size=(args.his_len, args.pred_len),
            grid_size=args.grid_size,
            features=args.features,
            target=args.target
        )
        
        dataset = WeatherDataset(
            root_path=args.data_path,
            flag=flag,
            size=(args.his_len, args.pred_len),
            grid_size=args.grid_size,
            features=args.features,
            target=args.target,
            scaler_std=train_dataset.scaler_std, 
            scaler_minmax=train_dataset.scaler_minmax, 
            scaler_robust=train_dataset.scaler_robust  
        )
    

    shuffle = True if flag == "train" else False
    
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=2 if args.num_workers > 0 else None,
        persistent_workers=True if args.num_workers > 0 else False,
        drop_last=True if flag == "train" else False
    )
    return data_loader, dataset

def traffic_data_provider(args, flag="train"):
    if flag == "train":
        train_dataset = TrafficDataset(
            root_path=args.data_path,
            flag="train",
            size=(args.his_len, args.pred_len),
            grid_size=args.grid_size,
        )
        dataset = train_dataset
    else:
        train_dataset = TrafficDataset(
            root_path=args.data_path,
            flag="train",
            size=(args.his_len, args.pred_len),
            grid_size=args.grid_size,
        )
        
        dataset = TrafficDataset(
            root_path=args.data_path,
            flag=flag,
            size=(args.his_len, args.pred_len),
            grid_size=args.grid_size,
            scaler=train_dataset.scaler  
        )
    

    shuffle = True if flag == "train" else False
    
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=2 if args.num_workers > 0 else None,
        persistent_workers=True if args.num_workers > 0 else False,
        drop_last=True if flag == "train" else False
    )
    return data_loader, dataset