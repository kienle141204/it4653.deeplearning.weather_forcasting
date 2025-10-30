from data_provider.data_loader import WeatherDataset
from torch.utils.data import DataLoader

def data_provider(args, flag="train"):
    dataset = WeatherDataset(
        root_path=args.data_path,
        flag=flag,
        size=(args.seq_len, args.pred_len),
        grid_size=args.grid_size
    )
    shuffle = True if flag == "train" else False
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers
    )
    return data_loader, dataset 
