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
import os 
import logging

# Bỏ qua tất cả các cảnh báo
warnings.filterwarnings('ignore')

# Suppress timm warnings
logging.getLogger('timm').setLevel(logging.ERROR)

# Login wandb only once (not in every import)
if not wandb.api.api_key:
    wandb.login(key=os.getenv('WANDB_KEY'))

# Enable optimizations
torch.backends.cudnn.benchmark = True  # Faster for fixed input sizes
torch.backends.cudnn.deterministic = False  # Faster but less reproducible

class Exp_Long_Term_Forecasting(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecasting, self).__init__(args)
        wandb.init(project="DL-project", name=f"{self.model.model_name}_{datetime.now().strftime('%Y-%m-%d %H:%M')}", config=vars(args))
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
        # Use AdamW with better weight decay
        optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=self.args.learning_rate, 
            weight_decay=1e-4,
            betas=(0.9, 0.999),
            eps=1e-8,
            fused=True  # Faster on modern GPUs
        ) 
        return optimizer
    
    def _create_scheduler(self, optimizer):
        # Use CosineAnnealingWarmRestarts for better convergence + ReduceLROnPlateau
        # Warmup + cosine annealing helps with accuracy
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=5
        )
        cosine_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
        plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=self.args.lr_patience,
            min_lr=1e-6
        )
        # Use plateau scheduler (more stable)
        return plateau_scheduler
    
    def _create_criterion(self):
        # Improved loss: MSE + Huber + L1 for better accuracy
        class CombinedLoss:
            def __init__(self):
                self.mse_loss = torch.nn.MSELoss()
                self.huber_loss = torch.nn.HuberLoss(delta=1.0)
                self.l1_loss = torch.nn.L1Loss()
            
            def __call__(self, pred, true):
                mse = self.mse_loss(pred, true)
                huber = self.huber_loss(pred, true)
                l1 = self.l1_loss(pred, true)
                # Weighted combination: 50% MSE, 30% Huber, 20% L1
                return 0.5 * mse + 0.3 * huber + 0.2 * l1
        
        return CombinedLoss()
    
    def vali(self, vali_data, vali_loader, criterion):
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
            # Use mixed precision for validation (faster)
            for index, seq_x, seq_y, seq_x_mark, seq_y_mark, sp  in vali_loader:
                seq_x = seq_x.to(self.device)
                seq_y = seq_y.to(self.device)
                
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        output = self.model(seq_x)
                else:
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
        
        # Use mixed precision for faster training (2x speedup)
        scaler = torch.cuda.amp.GradScaler() if self.args.use_amp else None

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
            
            for index, seq_x, seq_y, seq_x_mark, seq_y_mark, sp  in train_loader:
                iter_count += 1
                mask_true = schedule_sampling_exp(self.args, iter_count, train_steps)
                seq_x = seq_x.to(self.device)
                seq_y = seq_y.to(self.device)
                seq_x_mark = seq_x_mark.to(self.device)
                seq_y_mark = seq_y_mark.to(self.device)

                optimizer.zero_grad()
                
                # Mixed precision training
                if self.args.use_amp and scaler is not None:
                    with torch.cuda.amp.autocast():
                        output = self.model(seq_x, mask_true=mask_true, ground_truth=seq_y)
                        std_loss = criterion(output[:, :, std_cols_indices, :, :], seq_y[:, :, std_cols_indices, :, :]) if std_cols_indices else 0
                        minmax_loss = criterion(output[:, :, minmax_cols_indices, :, :], seq_y[:, :, minmax_cols_indices, :, :]) if minmax_cols_indices else 0
                        robust_loss = criterion(output[:, :, robust_cols_indices, :, :], seq_y[:, :, robust_cols_indices, :, :]) if robust_cols_indices else 0
                        tcc_loss = criterion(output[:, :, tcc_cols_indices, :, :], seq_y[:, :, tcc_cols_indices, :, :]) if tcc_cols_indices else 0
                        loss = std_loss + minmax_loss + robust_loss + tcc_loss
                    
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    output = self.model(seq_x, mask_true=mask_true, ground_truth=seq_y)
                    std_loss = criterion(output[:, :, std_cols_indices, :, :], seq_y[:, :, std_cols_indices, :, :]) if std_cols_indices else 0
                    minmax_loss = criterion(output[:, :, minmax_cols_indices, :, :], seq_y[:, :, minmax_cols_indices, :, :]) if minmax_cols_indices else 0
                    robust_loss = criterion(output[:, :, robust_cols_indices, :, :], seq_y[:, :, robust_cols_indices, :, :]) if robust_cols_indices else 0
                    tcc_loss = criterion(output[:, :, tcc_cols_indices, :, :], seq_y[:, :, tcc_cols_indices, :, :]) if tcc_cols_indices else 0
                    loss = std_loss + minmax_loss + robust_loss + tcc_loss
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()

                train_loss.append(loss.item())

            self.logger.info("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_start_time))
            train_loss = np.average(train_loss)
            
            # Validate every epoch (needed for early stopping and scheduler)
            # But skip test_loss calculation to save time (only calculate every 5 epochs)
            vali_loss = self.vali(vali_dataset, vali_loader, criterion)
            if epoch % 5 == 0 or epoch == 0:
                test_loss = self.vali(test_dataset, test_loader, criterion)
            else:
                test_loss = vali_loss  # Approximate
            
            wandb.log({"Train Loss": train_loss, "Validation Loss": vali_loss}, step=epoch)
            self.logger.info("Epoch: {0}, Steps: {1}, Lr: {2} | Train Loss: {3:.7f} Vali Loss: {4:.7f} Test Loss: {5:.7f}".format(
                epoch + 1, train_steps, optimizer.param_groups[0]['lr'], train_loss, vali_loss, test_loss))
            
            scheduler.step(vali_loss)
            
            # Early stopping
            if vali_loss < best_valid_loss:
                best_model = copy.deepcopy(self.model)
                best_valid_loss = vali_loss
                patience_counter = 0
                self.logger.info(f"New best model! Validation loss: {best_valid_loss:.7f}")
            else:
                patience_counter += 1
                self.logger.info(f"Early stopping counter: {patience_counter} out of {patience}")
                if patience_counter >= patience:
                    self.logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

        best_model_path = path + '/' + 'checkpoint.pth'
        torch.save(best_model.state_dict(), best_model_path)

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

                    outputs = outputs_inv.reshape(batch_size, his_len, height, width, n_features).transpose(0, 1, 4, 2, 3)
                    batch_y = batch_y_inv.reshape(batch_size, his_len, height, width, n_features).transpose(0, 1, 4, 2, 3)
                    

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
                        true_data=true[0, :, 0, 0, 0],
                        predicted_data=pred[0, :, 0, 0, 0],
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
        self.logger.info('Test - mse:{}, mae:{}, rmse:{}, mape: {}'.format(mse, mae, rmse, mape))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('Test - mse:{}, mae:{}, rmse:{}, mape: {}'.format(mse, mae, rmse, mape))
        f.write('\n')
        f.write('\n')
        f.close()


