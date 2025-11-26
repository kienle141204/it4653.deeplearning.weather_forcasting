from exp.exp_basic import Exp_Basic
from data_provider.data_factory import data_provider, traffic_data_provider
import torch
from torch import optim
import time 
import numpy as np
import copy
import warnings
import os 
from utils.metrics import metric
from utils.schedule_sampling import schedule_sampling_exp, reserve_schedule_sampling_exp
from utils.visualize import visualize, visualize_frame

# Bỏ qua tất cả các cảnh báo
warnings.filterwarnings('ignore')

class Exp_Long_Term_Forecasting(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecasting, self).__init__(args)
        

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        return model

    def _get_data(self, flag):
        data_loader, dataset = traffic_data_provider(self.args, flag)
        return data_loader, dataset
    
    def _create_optimizer(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=1e-5) 
        # optimizer = optim.Adam(self.model.parameters(), lr=self.args.learning_rate) 
        return optimizer
    
    def _create_scheduler(self, optimizer):
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=self.args.lr_patience,
            # verbose=True
        )
        return scheduler
    
    def _create_criterion(self):
        criterion = torch.nn.MSELoss()
        return criterion
    
    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for index, seq_x, seq_y  in vali_loader:
                seq_x = seq_x.to(self.device)
                seq_y = seq_y.to(self.device)
                # seq_x_mark = seq_x_mark.to(self.device)
                # seq_y_mark = seq_y_mark.to(self.device)

                output = self.model(seq_x)

                loss = criterion(output, seq_y)
                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss
        
    def train(self, setting):
        train_loader, train_dataset = self._get_data(flag="train")
        vali_loader, vali_dataset = self._get_data(flag="val")
        test_loader, test_dataset = self._get_data(flag="test")

        # col_names = train_dataset.col_names
        # std_cols = train_dataset.std_cols
        # minmax_cols = train_dataset.minmax_cols
        # robust_cols = train_dataset.robust_cols

        # std_cols_indices = [col_names.index(col) for col in std_cols]
        # minmax_cols_indices = [col_names.index(col) for col in minmax_cols] 
        # robust_cols_indices = [col_names.index(col) for col in robust_cols]
        # tcc_cols_indices = [col_names.index(col) for col in col_names if col not in std_cols + minmax_cols + robust_cols]

        _, seq_x_train, _ = next(iter(train_loader))
        _, seq_x_test, _= next(iter(test_loader))
        print(f"Shape x_train: {seq_x_train.shape}")
        print(f"Shape x_test: {seq_x_test.shape}")

        optimizer = self._create_optimizer()
        scheduler = self._create_scheduler(optimizer)
        criterion = self._create_criterion()

        patience = self.args.early_stop_patience  # Early stopping patience
        patience_counter = 0

        train_steps = len(train_loader)
        path = self.args.checkpoints + self.args.model + '/' + setting
        if not os.path.exists(path):
            os.makedirs(path)

        best_valid_loss = float('inf')
        best_model = None

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            self.model.train()
            train_loss = []

            epoch_start_time = time.time()
            for index, seq_x, seq_y  in train_loader:
                iter_count += 1
                mask_true = schedule_sampling_exp(self.args, iter_count, train_steps)
                if isinstance(mask_true, np.ndarray):
                    mask_true = torch.from_numpy(mask_true).float()
                mask_true = mask_true.to(self.device)
                # print(f"mask_true shape: {mask_true.shape}")
                seq_x = seq_x.to(self.device)
                seq_y = seq_y.to(self.device)
                # seq_x_mark = seq_x_mark.to(self.device)
                # seq_y_mark = seq_y_mark.to(self.device)

                # print(seq_x.shape, seq_y.shape)

                optimizer.zero_grad()
                output = self.model(seq_x, mask_true=mask_true, ground_truth=seq_y)

                loss = criterion(output, seq_y)

                train_loss.append(loss.item())
                loss.backward()
                optimizer.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_start_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_dataset, vali_loader, criterion)
            test_loss = self.vali(test_dataset, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            
            scheduler.step(vali_loss)
            
            # Early stopping
            if vali_loss < best_valid_loss:
                best_model = copy.deepcopy(self.model)
                best_valid_loss = vali_loss
                patience_counter = 0
                print(f"New best model! Validation loss: {best_valid_loss:.7f}")
            else:
                patience_counter += 1
                print(f"Early stopping counter: {patience_counter} out of {patience}")
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        best_model_path = path + '/' + 'checkpoint.pth'
        torch.save(best_model.state_dict(), best_model_path)

    def test(self, setting, test=0):
        folder_path = f'./results/{self.args.model}/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        test_loader, test_data = self._get_data(flag='test')
        if test:
            print("Loading Model")
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + self.args.model  + '/' + setting , 'checkpoint.pth')))
        
        preds = []
        trues = []
        self.model.eval()
        with torch.no_grad():
            for i, (index, batch_x, batch_y) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                # batch_x_mark = batch_x_mark.float().to(self.device)
                # batch_y_mark = batch_y_mark.float().to(self.device)

                outputs = self.model(batch_x)

                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                if self.args.inverse:
                    shape = outputs.shape
                    
                    batch_size, his_len, n_features, height, width = shape
                    outputs_reshaped = outputs.transpose(0, 1, 3, 4, 2).reshape(-1, n_features)
                    batch_y_reshaped = batch_y.transpose(0, 1, 3, 4, 2).reshape(-1, n_features)

                    outputs_inv = test_data.inverse_transform(outputs_reshaped)
                    batch_y_inv = test_data.inverse_transform(batch_y_reshaped)

                    outputs = outputs_inv.reshape(batch_size, his_len, height, width, n_features).transpose(0, 1, 4, 2, 3)
                    batch_y = batch_y_inv.reshape(batch_size, his_len, height, width, n_features).transpose(0, 1, 4, 2, 3)
                    

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)

                if i % 10 == 0:
                    visualize(
                        historical_data=batch_x[0,:,0, 0, 0].detach().cpu().numpy(),
                        true_data=batch_y[0,:,0, 0, 0],
                        predicted_data=outputs[0,:,0, 0, 0],
                        title=f"Test Sample {i} - Traffic - Forecasting",
                        xlabel="Time Steps",
                        ylabel="Value",
                        save_path=f"./results/{self.args.model}/{setting}/traffic_sample_{i}.png"
                    )

                    visualize_frame(
                        historical_data=batch_x[0, :, 0, :, :].detach().cpu().numpy(),
                        true_data=batch_y[0, :, 0, :, :],   
                        predicted_data=outputs[0, :, 0, :, :],
                        title=f"Test Sample {i} - Traffic - Spatio-Temporal Forecasting",
                        xlabel="Future Time Steps",
                        save_path=f"./results/{self.args.model}/{setting}/traffic_spatiotemporal_sample_{i}.png"
                    )
                    
        
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)

        
        mae, mse, rmse, mape= metric(preds, trues)
        print('mse:{}, mae:{}, rmse:{}, mape: {}'.format(mse, mae, rmse, mape))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, rmse:{}, mape: {}'.format(mse, mae, rmse, mape))
        f.write('\n')
        f.write('\n')
        f.close()

# def to_numpy(tensor_or_array):
#     if isinstance(tensor_or_array, torch.Tensor):
#         return tensor_or_array.detach().cpu().numpy()
#     elif isinstance(tensor_or_array, np.ndarray):
#         return tensor_or_array
#     else:
#         return np.array(tensor_or_array)


