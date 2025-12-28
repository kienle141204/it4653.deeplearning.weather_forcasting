from exp.exp_basic import Exp_Basic
from data_provider.data_factory import data_provider
import torch
from torch import optim
import time 
from datetime import datetime
import numpy as np
import copy
import warnings
import os 
from utils.metrics import metric
from utils.schedule_sampling import schedule_sampling_exp, reserve_schedule_sampling_exp
from utils.visualize import visualize, visualize_frame
from utils.logger import getlogger
import wandb 
import logging

# Bỏ qua tất cả các cảnh báo
warnings.filterwarnings('ignore')

# Suppress timm warnings
logging.getLogger('timm').setLevel(logging.ERROR)

# Login wandb only once (not in every import)
# Check if running in Kaggle environment (disable wandb if so)
is_kaggle = os.environ.get('KAGGLE_KERNEL_RUN_TYPE') is not None

# Set wandb to non-interactive mode for Kaggle/CI environments
os.environ['WANDB_SILENT'] = 'true'
os.environ['WANDB_DISABLE_CODE'] = 'true'
os.environ['WANDB_CONSOLE'] = 'off'

# For Kaggle or if explicitly disabled, skip wandb
if is_kaggle or os.environ.get('WANDB_MODE') == 'disabled':
    os.environ['WANDB_MODE'] = 'disabled'
    print("WandB is disabled (Kaggle environment or WANDB_MODE=disabled)")
else:
    # Force login with the specified key (relogin=True to override cached credentials)
    # Set API key directly to avoid interactive prompt
    try:
        os.environ['WANDB_API_KEY'] = 'cc3ae1cab3d94989520cf4c8164d37c29744f1b2'
        # Use _disable_stats=True to avoid any prompts
        wandb.login(key='cc3ae1cab3d94989520cf4c8164d37c29744f1b2', relogin=True)
    except Exception as e:
        # If login fails, disable wandb to avoid blocking
        print(f"Warning: wandb login failed: {e}. Continuing without wandb.")
        os.environ['WANDB_MODE'] = 'disabled'

# Enable optimizations
torch.backends.cudnn.benchmark = True  # Faster for fixed input sizes
torch.backends.cudnn.deterministic = False  # Faster but less reproducible

class Exp_Long_Term_Forecasting(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecasting, self).__init__(args)
        # Only init wandb if not disabled
        if os.environ.get('WANDB_MODE') != 'disabled':
            try:
                wandb.init(project="DL-project", name=f"{self.model.model_name}_{datetime.now().strftime('%Y-%m-%d %H:%M')}", config=vars(args))
            except Exception as e:
                print(f"Warning: wandb.init failed: {e}. Continuing without wandb.")
                os.environ['WANDB_MODE'] = 'disabled'
        self.logger = getlogger(logpath='./logs/experiment.log', name='Experiment')
        self.logger.info(f"Model: {self.model}")
        

    def _build_model(self):
        # Get dataset info to determine channel groups
        train_loader, train_dataset = self._get_data(flag='train')
        col_names = train_dataset.col_names
        std_cols = train_dataset.std_cols
        minmax_cols = train_dataset.minmax_cols
        robust_cols = train_dataset.robust_cols
        tcc_cols = train_dataset.tcc_cols
        
        std_cols_indices = [col_names.index(col) for col in std_cols]
        minmax_cols_indices = [col_names.index(col) for col in minmax_cols]
        robust_cols_indices = [col_names.index(col) for col in robust_cols]
        tcc_cols_indices = [col_names.index(col) for col in tcc_cols]
        
        self.args.std_cols_indices = std_cols_indices
        self.args.minmax_cols_indices = minmax_cols_indices
        self.args.robust_cols_indices = robust_cols_indices
        self.args.tcc_cols_indices = tcc_cols_indices
        
        self.args.num_std = len(std_cols_indices)
        self.args.num_minmax = len(minmax_cols_indices)
        self.args.num_robust = len(robust_cols_indices)
        self.args.num_tcc = len(tcc_cols_indices)

        model = self.model_dict[self.args.model](self.args).float()

        return model

    def _get_data(self, flag):
        data_loader, dataset = data_provider(self.args, flag)
        return data_loader, dataset
    
    def _create_optimizer(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=1e-5) 
        # optimizer = optim.Adam(self.model.parameters(), lr=self.args.learning_rate) 
        return optimizer
    
    def _create_scheduler(self, optimizer):
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=self.args.lr_patience,
            # verbose=True
        )
        return scheduler
    
    def _create_criterion(self):
        criterion = torch.nn.MSELoss()
        return criterion
    
    def vali(self, vali_data, vali_loader, criterion, debug=False):
        col_names = vali_data.col_names
        std_cols = vali_data.std_cols
        minmax_cols = vali_data.minmax_cols
        robust_cols = vali_data.robust_cols

        std_cols_indices = [col_names.index(col) for col in std_cols]
        minmax_cols_indices = [col_names.index(col) for col in minmax_cols] 
        robust_cols_indices = [col_names.index(col) for col in robust_cols]
        tcc_cols_indices = [col_names.index(col) for col in col_names if col not in std_cols + minmax_cols + robust_cols]

        total_loss = []
        self.model.eval()
        
        with torch.no_grad():
            for i, (index, seq_x, seq_y, seq_x_mark, seq_y_mark, sp) in enumerate(vali_loader):
                seq_x = seq_x.to(self.device)
                seq_y = seq_y.to(self.device)
                seq_x_mark = seq_x_mark.to(self.device)
                seq_y_mark = seq_y_mark.to(self.device)

                output = self.model(seq_x)

                std_loss = criterion(output[:, :, std_cols_indices, :, :], seq_y[:, :, std_cols_indices, :, :]) if std_cols_indices else 0
                minmax_loss = criterion(output[:, :, minmax_cols_indices, :, :], seq_y[:, :, minmax_cols_indices, :, :]) if minmax_cols_indices else 0
                robust_loss = criterion(output[:, :, robust_cols_indices, :, :], seq_y[:, :, robust_cols_indices, :, :]) if robust_cols_indices else 0
                tcc_loss = criterion(output[:, :, tcc_cols_indices, :, :], seq_y[:, :, tcc_cols_indices, :, :]) if tcc_cols_indices else 0

                loss = std_loss + minmax_loss + robust_loss + tcc_loss
                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss
        
    def train(self, setting):
        train_loader, train_dataset = self._get_data(flag="train")
        vali_loader, vali_dataset = self._get_data(flag="val")
        test_loader, test_dataset = self._get_data(flag="test")

        col_names = train_dataset.col_names
        std_cols = train_dataset.std_cols
        minmax_cols = train_dataset.minmax_cols
        robust_cols = train_dataset.robust_cols

        std_cols_indices = [col_names.index(col) for col in std_cols]
        minmax_cols_indices = [col_names.index(col) for col in minmax_cols] 
        robust_cols_indices = [col_names.index(col) for col in robust_cols]
        tcc_cols_indices = [col_names.index(col) for col in col_names if col not in std_cols + minmax_cols + robust_cols]

        _, seq_x_train, _, _, _, _ = next(iter(train_loader))
        _, seq_x_test, _, _, _, _ = next(iter(test_loader))
        self.logger.info(f"Shape x_train: {seq_x_train.shape}")
        self.logger.info(f"Shape x_test: {seq_x_test.shape}")

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
        best_model_state = None
        patience_counter = 0
        patience = self.args.early_stop_patience

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            self.model.train()
            train_loss = []

            epoch_start_time = time.time()
            for index, seq_x, seq_y, seq_x_mark, seq_y_mark, sp  in train_loader:
                iter_count += 1
                mask_true = schedule_sampling_exp(self.args, iter_count, train_steps)
                seq_x = seq_x.to(self.device)
                seq_y = seq_y.to(self.device)
                seq_x_mark = seq_x_mark.to(self.device)
                seq_y_mark = seq_y_mark.to(self.device)

                # print(seq_x.shape, seq_y.shape)

                optimizer.zero_grad()
                output = self.model(seq_x, mask_true=mask_true, ground_truth=seq_y)

                std_loss = criterion(output[:, :, std_cols_indices, :, :], seq_y[:, :, std_cols_indices, :, :]) if std_cols_indices else 0
                minmax_loss = criterion(output[:, :, minmax_cols_indices, :, :], seq_y[:, :, minmax_cols_indices, :, :]) if minmax_cols_indices else 0
                robust_loss = criterion(output[:, :, robust_cols_indices, :, :], seq_y[:, :, robust_cols_indices, :, :]) if robust_cols_indices else 0
                tcc_loss = criterion(output[:, :, tcc_cols_indices, :, :], seq_y[:, :, tcc_cols_indices, :, :]) if tcc_cols_indices else 0

                loss = std_loss + minmax_loss + robust_loss + tcc_loss

                train_loss.append(loss.item())
                loss.backward()
                optimizer.step()

            self.logger.info("[Train] - epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_start_time))
            train_loss = np.average(train_loss)
            
            # Debug mode cho epoch đầu tiên
            debug_mode = (epoch == 0)
            vali_loss = self.vali(vali_dataset, vali_loader, criterion, debug=debug_mode)
            test_loss = self.vali(test_dataset, test_loader, criterion, debug=debug_mode)

            # Only log to wandb if not disabled
            if os.environ.get('WANDB_MODE') != 'disabled':
                try:
                    wandb.log({"train loss": train_loss, "vali loss": vali_loss}, step=epoch)
                except:
                    pass

            if epoch % 5 == 0:
                self.logger.info("[Vali] - epoch: {0}, steps: {1}, lr: {2} | [train loss: {3:.7f}] - [vali loss: {4:.7f}] - [test loss: {5:.7f}]".format(
                    epoch + 1, train_steps, optimizer.param_groups[0]['lr'], train_loss, vali_loss, test_loss))
            else:
                self.logger.info("[Vali] - epoch: {0}, steps: {1}, lr: {2} | [train loss: {3:.7f}] - [vali loss: {4:.7f}]".format(
                    epoch + 1, train_steps, optimizer.param_groups[0]['lr'], train_loss, vali_loss))
            
            scheduler.step(vali_loss)
            
            # Early stopping - lưu model tốt nhất
            if vali_loss < best_valid_loss:
                # Lưu state_dict thay vì deepcopy để tránh vấn đề với buffer
                best_valid_loss = vali_loss
                best_model_state = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
                self.logger.info(f"New best model! Validation loss: {best_valid_loss:.7f}")
            else:
                patience_counter += 1
                self.logger.info(f"Early stopping counter: {patience_counter} out of {patience}")
                if patience_counter >= patience:
                    self.logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

        # Lưu model tốt nhất
        best_model_path = path + '/' + 'checkpoint.pth'
        if best_model_state is not None:
            torch.save(best_model_state, best_model_path)
            self.logger.info(f"Model saved to {best_model_path}")
        else:
            # Fallback: lưu model hiện tại nếu không có best_model
            torch.save(self.model.state_dict(), best_model_path)
            self.logger.warning(f"No best model found, saving current model to {best_model_path}")

    def test(self, setting, test=0):
        folder_path = f'./results/{self.args.model}/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        test_loader, test_data = self._get_data(flag='test')
        if test:
            self.logger.info("Loading Model")
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + self.args.model  + '/' + setting , 'checkpoint.pth')))
        
        preds = []
        trues = []
        self.model.eval()
        with torch.no_grad():
            for i, (index, batch_x, batch_y, batch_x_mark, batch_y_mark, sq) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

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

                    outputs_inv = outputs_inv.reshape(batch_size, his_len, height, width, n_features).transpose(0, 1, 4, 2, 3)
                    batch_y_inv = batch_y_inv.reshape(batch_size, his_len, height, width, n_features).transpose(0, 1, 4, 2, 3)
                    
                    pred = outputs_inv
                    true = batch_y_inv
                else:
                    pred = outputs
                    true = batch_y

                preds.append(pred)
                trues.append(true)
                if i % 10 == 0:
                    # print("shape")
                    # print(batch_x_mark.shape, batch_y_mark.shape)
                    # print(batch_x_mark[0, :, :].shape, batch_y_mark[0, :, :].shape)
                    # print(batch_x_mark[0, 0, :].detach().cpu().numpy())
                    visualize(
                        historical_data=batch_x[0, :, 0, 0, 0].detach().cpu().numpy(),
                        true_data=batch_y[0, :, 0, 0, 0],
                        predicted_data=outputs[0, :, 0, 0, 0],
                        x_mark=batch_x_mark[0, :, :].detach().cpu().numpy(),
                        y_mark=batch_y_mark[0, :, :].detach().cpu().numpy(),
                        title=f"{self.model.model_name}-Test Sample {i}-Weather Forecasting",
                        xlabel="Time Steps",
                        ylabel="Value",
                        save_path=f"./results/{self.args.model}/{setting}/forecast_sample_{i}.png"
                    )
                    visualize_frame(
                        historical_data=batch_x[0, :, 0, :, :].detach().cpu().numpy(),
                        true_data=batch_y[0, :, 0, :, :],   
                        predicted_data=outputs[0, :, 0, :, :],
                        x_mark=batch_x_mark[0, :, :].detach().cpu().numpy(),
                        y_mark=batch_y_mark[0, :, :].detach().cpu().numpy(),
                        title=f"{self.model.model_name}-Test Sample {i}-Weather Forecasting",
                        # xlabel="Future Time Steps",
                        save_path=f"./results/{self.args.model}/{setting}/weather_spatiotemporal_sample_{i}.png"
                    )
        
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)

        
        mae, mse, rmse, mape= metric(preds, trues)
        self.logger.info('Test - [mse: {}], [mae: {}], [rmse: {}], [mape: {}]'.format(mse, mae, rmse, mape))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('Test - [mse: {}], [mae: {}], [rmse: {}], [mape: {}]'.format(mse, mae, rmse, mape))
        f.write('\n')
        f.write('\n')
        f.close()


