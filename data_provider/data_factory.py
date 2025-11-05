from data_provider.data_loader import WeatherDataset
from torch.utils.data import DataLoader

def data_provider(args, flag="train"):
    if flag == "train":
        train_dataset = WeatherDataset(
            root_path=args.data_path,
            flag="train",
            size=(args.seq_len, args.pred_len),
            grid_size=args.grid_size,
            features=args.features,
            target=args.target
        )
        dataset = train_dataset
    else:
        train_dataset = WeatherDataset(
            root_path=args.data_path,
            flag="train",
            size=(args.seq_len, args.pred_len),
            grid_size=args.grid_size,
            features=args.features,
            target=args.target
        )
        
        dataset = WeatherDataset(
            root_path=args.data_path,
            flag=flag,
            size=(args.seq_len, args.pred_len),
            grid_size=args.grid_size,
            features=args.features,
            target=args.target,
            scaler_std=train_dataset.scaler_std, 
            scaler_minmax=train_dataset.scaler_minmax, 
            scaler_robust=train_dataset.scaler_robust  
        )
    

    shuffle = False
    
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers
    )
    return data_loader, dataset