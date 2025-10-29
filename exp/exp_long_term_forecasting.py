from exp.exp_basic import Exp_Basic
from data_provider.data_factory import data_provider
import torch
from torch import optim
import time 
import numpy as np
import copy
import warnings

# Bỏ qua tất cả các cảnh báo
warnings.filterwarnings('ignore')

class Exp_Long_Term_Forecasting(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecasting, self).__init__(args)
        

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        return model

    def _get_data(self, frag):
        data_loader, dataset = data_provider(self.args, frag)
        return data_loader, dataset
    
    def _create_optimizer(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=1e-5) 
        return optimizer
    
    def _create_scheduler(self, optimizer):
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        return scheduler
    
    def _create_criterion(self):
        criterion = torch.nn.MSELoss()
        return criterion
    
    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for index, seq_x, seq_y, seq_x_mark, seq_y_mark, sp  in vali_loader:
                seq_x = seq_x.to(self.device)
                seq_y = seq_y.to(self.device)
                seq_x_mark = seq_x_mark.to(self.device)
                seq_y_mark = seq_y_mark.to(self.device)

                output = self.model(seq_x)
                loss = criterion(output, seq_y)
                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss
        
    def train(self):
        train_loader, train_dataset = self._get_data(frag="train")
        vali_loader, vali_dataset = self._get_data(frag="val")
        test_loader, test_dataset = self._get_data(frag="test")

        optimizer = self._create_optimizer()
        scheduler = self._create_scheduler(optimizer)
        criterion = self._create_criterion()

        patience = 3  # Early stopping patience
        patience_counter = 0

        train_steps = len(train_loader)
        path = self.args.checkpoints + self.args.model

        best_valid_loss = float('inf')
        best_model = None

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            self.model.train()
            train_loss = []

            epoch_start_time = time.time()
            for index, seq_x, seq_y, seq_x_mark, seq_y_mark, sp  in train_loader:
                iter_count += 1
                seq_x = seq_x.to(self.device)
                seq_y = seq_y.to(self.device)
                seq_x_mark = seq_x_mark.to(self.device)
                seq_y_mark = seq_y_mark.to(self.device)

                # print(seq_x.shape, seq_y.shape)

                optimizer.zero_grad()
                output = self.model(seq_x)
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



