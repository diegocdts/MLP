import os

import numpy as np
import torch
from sklearn.model_selection import KFold
from torch import nn, optim
from torch.utils.data import Subset, DataLoader

from data.data import input_target_names, load_data, TraceDataset, load_file, zscore, compare
from model.model import MLP


class CrossValidation:

    def __init__(self, data_info, model_info, train_info):
        i_names, t_names = input_target_names(data_info["blended"], data_info["single"], 0, data_info["n_files"])
        i_data, t_data, self.i_mean, self.i_std = load_data(i_names, t_names, data_info["n_receivers"])
        self.dataset = TraceDataset(i_data, t_data)
        self.model_io = i_data.shape[-1] #6001
        print(i_data.shape)
        self.data_info = data_info
        self.model_info = model_info
        self.train_info = train_info

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.lowest_loss = float('inf')
        self.outputs_path = data_info["outputs_path"].replace('_XX_', f'_{model_info["name"]}_')
        self.predictions_path = os.path.join(self.outputs_path, 'predictions')
        self.model_path = os.path.join(self.outputs_path, 'model_weights.pth')

        if not os.path.exists(self.outputs_path):
            os.mkdir(self.outputs_path)

        if not os.path.exists(self.predictions_path):
            os.mkdir(self.predictions_path)

    def train(self):
        k_fold = KFold(n_splits=self.train_info["k_fold"], shuffle=False)
        print('Preparing to start cross validation')
        for fold, (idx_train, idx_val) in enumerate(k_fold.split(self.dataset)):
            t_subset = Subset(self.dataset, idx_train)
            v_subset = Subset(self.dataset, idx_val)

            t_loader = DataLoader(t_subset, batch_size=self.train_info["batch_size"], shuffle=True)
            v_loader = DataLoader(v_subset, batch_size=self.train_info["batch_size"], shuffle=False)

            model = MLP(io_units=self.model_io, hidden_units=self.model_info["hidden_units"])
            model.to(self.device)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=self.train_info['lr'])

            train_losses, val_losses = [], []
            for epoch in range(self.train_info["n_epochs"]):
                model.train()
                train_loss = 0
                for inputs, targets in t_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                train_epoch_loss = train_loss / len(t_loader)

                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for inputs, targets in v_loader:
                        inputs, targets = inputs.to(self.device), targets.to(self.device)
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                        val_loss += loss.item()
                    val_epoch_loss = val_loss / len(v_loader)

                self.log(fold, epoch, train_epoch_loss, val_epoch_loss)
                train_losses.append(train_epoch_loss)
                val_losses.append(val_epoch_loss)

            if sum(val_losses) / len(val_losses) < self.lowest_loss:
                self.lowest_loss = sum(val_losses) / len(val_losses)
                torch.save(model.state_dict(), self.model_path)
            self.write_losses(train_losses, val_losses)

    def predict(self, input_path):
        input_data = zscore(load_file(input_path), self.i_mean, self.i_std)
        test_dataset = TraceDataset(input_data)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        model = MLP(io_units=self.model_io, hidden_units=self.model_info["hidden_units"])
        model.to(self.device)

        model.load_state_dict(torch.load(self.model_path, weights_only=True))
        model.eval()

        predictions = []
        with torch.no_grad():
            for trace in test_loader:
                trace = trace.to(self.device)
                prediction = model(trace)
                prediction = prediction.cpu().numpy()
                predictions.append(prediction)
        predictions = np.array(predictions)
        predictions = predictions.reshape(predictions.shape[0] * predictions.shape[1], predictions.shape[2])
        suffix_target = input_path[-20:]
        suffix_prediction = input_path[-20:-4]

        target_path = os.path.join(self.data_info["single"], f'SISMO_SINGLE{suffix_target}')
        predictions_path = os.path.join(self.predictions_path, f'PREDICTION{suffix_prediction}')
        np.save(predictions_path, predictions)

        compare(input_path, target_path, predictions_path, self.predictions_path)


    def log(self, fold, epoch, train_epoch_loss, val_epoch_loss):
        print(f'FOLD {fold+1}/{self.train_info["k_fold"]} - EPOCH {epoch+1}/{self.train_info["n_epochs"]} - '
              f'LOSS train {train_epoch_loss} valid {val_epoch_loss}')

    def write_losses(self, train_losses, val_losses):
        np.savetxt(os.path.join(self.outputs_path, 'train_losses.csv'), np.array(train_losses), fmt='%.5f')
        np.savetxt(os.path.join(self.outputs_path, 'val_losses.csv'), np.array(val_losses), fmt='%.5f')