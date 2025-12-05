import os
import sys
import copy
from functools import partial
import math
import warnings
import bisect
import numpy as np
import xarray as xr
import dask
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pytorch_lightning as pl
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from pytorch_lightning.core.mixins import HyperparametersMixin
from pytorch_lightning.profilers import SimpleProfiler

from global_variables import *
from functions import *

sys.tracebacklimit = None


class TransposeBatchNorm(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.bn = nn.BatchNorm1d(hidden_size)

    def forward(self, x):
        # x shape: (batch, seq_len, hidden_size)
        x = x.transpose(1, 2)  # (batch, hidden_size, seq_len)
        x = self.bn(x)
        x = x.transpose(1, 2)  # (batch, seq_len, hidden_size)
        return x
    

class DummyLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
    
class MultiHeadOutput(nn.Module):
    def __init__(self, hidden_size, n_targets):
        super().__init__()
        self.n_targets = n_targets
        self.class_head = nn.Linear(hidden_size, 1)
        self.mu_head = nn.Linear(hidden_size, 1)
        if n_targets == 3:
            self.logvar_head = nn.Linear(hidden_size, 1)
            nn.init.constant_(self.logvar_head.bias, -2.0)

    def forward(self, x):
        # x: (batch, seq_len, hidden_size)
        binary_out = self.class_head(x)
        mu = self.mu_head(x)
        output = torch.cat([binary_out, mu], dim=-1)
        if self.n_targets == 3:
            log_var = self.logvar_head(x)
            output = torch.cat([output, log_var], dim=-1)
        return output

class ReconstructionDataset(HyperparametersMixin, Dataset):
    def __init__(self, data_arr, features, targets, spinup_length=24, prediction_length=0, prediction_start_idx=0):
        '''
        prediction_length: If 0 then predict all values (besides those in spinup)
        '''
        super().__init__()
        self.save_hyperparameters()
        self.x = torch.nan_to_num(torch.as_tensor(self.hparams.data_arr.sel(variable = self.hparams.features).transpose('sample', 'time', 'variable').values).float(), 0)
        self.y = torch.as_tensor(self.hparams.data_arr.sel(variable = self.hparams.targets).transpose('sample', 'time', 'variable').values).float()
        self.calculate_num_temporal_samples()

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        if self.hparams.prediction_length == 0:
            prediction_start = self.hparams.prediction_start_idx+self.hparams.spinup_length
            prediction_end = self.x.shape[1]
            spinup_start = 0
        else:
            spinup_start = torch.randint(low=0, high=self.temporal_samples-1, size=(1, 1)).detach()
            prediction_start = spinup_start+self.hparams.spinup_length
            prediction_end = prediction_start+self.hparams.prediction_length
        x = self.x[idx, spinup_start:prediction_end, :]
        y = self.y[idx, prediction_start:prediction_end, :]
        return x, y
    
    def calculate_num_temporal_samples(self):
        self.temporal_samples = self.x.shape[1] - self.hparams.spinup_length - self.hparams.prediction_length - self.hparams.prediction_start_idx + 1
    
    def change_prediction_length(self, new_prediction_length):
        self.hparams.prediction_length = new_prediction_length
        self.calculate_num_temporal_samples()

    def change_prediction_start(self, new_prediction_start_idx):
        self.hparams.prediction_start_idx = new_prediction_start_idx
        self.calculate_num_temporal_samples()

    def normalize(self, mean, std):
        self.x = (self.x-mean)/std

    def unnormalize(self, mean, std):
        self.x = (self.x*std)+mean

    def normalize_targets(self, mean, std):
        self.y = (self.y-mean)/std

    def unnormalize_targets(self, mean, std):
        self.y = (self.y*std)+mean


class MultipleDataset(Dataset):
    def __init__(self, datasets):
        super().__init__()
        self.datasets = datasets
        if len(datasets):
            self.calculate_idcs()
            self.weights = self._compute_weights()
        else:
            warnings.warn("Warning: Dataset length is 0. This is only acceptable if this instance is a temporary placeholder.")

    def calculate_idcs(self):
        self.datasets_lengths = [len(ds) for ds in self.datasets]
        self.datasets_start_idx = [0]
        for el in self.datasets_lengths[:-1]:
            self.datasets_start_idx.append(el+self.datasets_start_idx[-1])

    def __len__(self):
        return sum(self.datasets_lengths)
    
    def __getitem__(self, idx):
        ds_index = bisect.bisect_right(self.datasets_start_idx, idx) - 1
        index_inside_ds = idx - self.datasets_start_idx[ds_index]
        return self.datasets[ds_index][index_inside_ds]
    
    def normalize(self, mean, std):
        for ds in self.datasets:
            ds.normalize(mean, std)

    def unnormalize(self, mean, std):
        for ds in self.datasets:
            ds.unnormalize(mean, std)

    def normalize_targets(self, mean, std):
        for ds in self.datasets:
            ds.normalize_targets(mean, std)

    def unnormalize_targets(self, mean, std):
        for ds in self.datasets:
            ds.unnormalize_targets(mean, std)


    def _compute_weights(self):
        weights = []
        for dataset in self.datasets:
            data = dataset.y  # Assuming each dataset has a `.y` attribute
            dataset_weights = (data.sum(axis=1) > 0).float() + 1.0
            weights.append(dataset_weights)
        return torch.cat(weights).squeeze()

activ_dict = {
    'ReLU': nn.ReLU(),
    'Tanh': nn.Tanh(),
    'Sigmoid': nn.Sigmoid(),
    'Softplus': partial(nn.Softplus(beta=1)).func,
    None: DummyLayer()
}            

class LightningLSTM(pl.LightningModule):
    def __init__(self, 
                 input_size, 
                 hidden_size, 
                 loss_f,
                 output_size=1, 
                 num_layers=1, 
                 linear = True,
                 lstm_activ_f='Tanh', 
                 linear_activ_f='ReLU', 
                 dynamic_loss=False, 
                 dropout=0, 
                 lr=0.1, 
                 spinup_length=24, 
                 ba_as_predictor=False, 
                 teacher_forcing_ratio=0.95,
                 seperate_heads=False,
                 seed=SEED):
        ## input_size = number of features (or variables) in the data. 
        ## hidden_size = this determines the dimension of the output
        ##               in other words, if we set hidden_size=1, then we have 1 output node
        ##               if we set hidden_size=50, then we have 50 output nodes (that could then be 50 input
        ##               nodes to a subsequent fully connected neural network).
        super().__init__()
        pl.seed_everything(seed=seed)
        self.save_hyperparameters()

        if dynamic_loss:
            self.hparams.val_loss_f = copy.deepcopy(self.hparams.loss_f)
            self.hparams.val_loss_f.keywords['alpha'] = 0.5
        else:
            self.hparams.val_loss_f = self.hparams.loss_f



        self.lstm_activ_f = activ_dict[lstm_activ_f]
        self.linear_activ_f = activ_dict[linear_activ_f]

        if isinstance(loss_f, partial):
            name = loss_f.func
        else:
            name = loss_f

        if name == Binary_and_MSELoss:
            self.last_activ_f = self._activation_Binary_MSE
        else:
            self.last_activ_f = self._activation_f

        if self.hparams.linear:
            self.lstm = nn.LSTM(input_size=self.hparams.input_size+self.hparams.ba_as_predictor, hidden_size=self.hparams.hidden_size, num_layers=self.hparams.num_layers, batch_first=True, dropout=self.hparams.dropout) 
            self.batchnorm = TransposeBatchNorm(self.hparams.hidden_size)
            self.dropout = nn.Dropout(p=self.hparams.dropout)
            if self.hparams.seperate_heads:
                self.fc = MultiHeadOutput(self.hparams.hidden_size, self.hparams.output_size)
            else:
                self.fc = nn.Linear(self.hparams.hidden_size, self.hparams.output_size)
            self.model_sequence = nn.Sequential(
                self.batchnorm,
                self.lstm_activ_f,
                self.dropout,
                self.fc,
                self.linear_activ_f
                )
        else:
            self.lstm = nn.LSTM(input_size=self.hparams.input_size+self.hparams.ba_as_predictor, hidden_size=self.hparams.hidden_size, num_layers=self.hparams.num_layers, batch_first=True, dropout=self.hparams.dropout, proj_size=self.hparams.output_size)
            self.model_sequence = nn.Sequential(
                self.lstm_activ_f
                )
            

    def _activation_Binary_MSE(self, outputs):
        if self.hparams.linear:
            outputs[:, :, 1] = self.linear_activ_f(outputs[:, :, 1])
        else:
            outputs[:, :, 1] = self.lstm_activ_f(outputs[:, :, 1])
        return outputs
    
    def _activation_f(self, outputs):
        if self.hparams.linear:
            return self.linear_activ_f(outputs)
        else:
            return self.lstm_activ_f(outputs)


    @torch.jit.export
    def forward(self, x, y=None):
        """
        x: (batch_size, seq_length, n_features) - The input features.
        y: (batch_size, seq_length, 1) - The true burned area values (for teacher forcing during training).
        """

        if self.hparams.ba_as_predictor:
            batch_size, seq_len, num_features = x.shape
            device = x.device
            burned_area = torch.zeros(batch_size, 1, device=device)     
            h_0, c_0 = torch.zeros(self.hparams.num_layers, batch_size, self.hparams.hidden_size, device=device), torch.zeros(self.hparams.num_layers, batch_size, self.hparams.hidden_size, device=device)
            outputs = torch.zeros(batch_size, seq_len, 1, device=device)
            # Loop over timesteps

            
            for t in range(seq_len):
                # Concatenate the features at time t with the burned area from previous timestep
                # x[:, t] shape: (batch_size, 36), burned_area shape: (batch_size, 1)
                input_t = torch.cat([x[:, t], burned_area], dim=-1).unsqueeze(1)  # Shape: (batch_size, 1, 37)
                
                # One LSTM step
                lstm_out, (h_0, c_0) = self.lstm(input_t, (h_0, c_0))  # out shape: (batch_size, 1, hidden_dim)
                if not self.hparams.linear:
                    prediction = self.last_activ_f(lstm_out)
                else:
                    lstm_out = self.lstm_activ_f(lstm_out)
                    if batch_size>1:
                        lstm_out = self.batchnorm(lstm_out.permute(0, 2, 1)).permute(0, 2, 1)     
                    prediction = self.last_activ_f(self.fc(lstm_out))  # Shape: (batch_size, 1, 1)
                outputs[:, t] = prediction.squeeze(1)
                
                # For the next timestep, decide whether to use teacher forcing
                if y is not None:
                    if torch.rand(1).item() < self.hparams.teacher_forcing_ratio:
                        burned_area = y[:, t, :]  # Use ground truth
                    else:
                        burned_area = prediction.squeeze(1)  # Use model prediction
                else:
                    burned_area = prediction.squeeze(1)  # Use model prediction at inference

        else:
            lstm_out, _ = self.lstm(x)
            outputs = self.model_sequence(lstm_out)
            
        return outputs[:, self.hparams.spinup_length:, :]
    
    def configure_optimizers(self): # this configures the optimizer we want to use for backpropagation.
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=1e-5, fused=True)

        scheduler = {
            'scheduler': ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=2, min_lr=1e-8),
            'monitor': 'val_loss',
            'interval': 'epoch'
        }

        return [optimizer], [scheduler]

    def on_train_epoch_start(self):
        # Dynamically adjust alpha
        if self.hparams.dynamic_loss:
            self.hparams.loss_f.keywords['alpha'] = max(0.05, self.hparams.loss_f.keywords['alpha']*0.9) # Decay alpha over time

    def on_train_epoch_end(self):
        if self.hparams.ba_as_predictor:
            self.hparams.teacher_forcing_ratio -= 0.1

    
    def training_step(self, batch, batch_idx): # take a step during gradient descent.
        #features = torch.cat([sample[0] for sample in batch], dim=0)
        #labels = torch.cat([sample[1] for sample in batch], dim=0)
        features, labels = batch # collect input
        predictions = self.forward(features) # run input through the neural network
        loss, log_params = self.hparams.loss_f(predictions, labels)
        log_params = {f'train_{key}': value for key, value in log_params.items()}
        self.log_dict(log_params, prog_bar=False, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        features, labels =  batch # collect input
        predictions = self.forward(features) # run input through the neural network
        loss, log_params = self.hparams.val_loss_f(predictions, labels)
        log_params = {f'val_{key}': value for key, value in log_params.items()}
        self.log_dict(log_params, prog_bar=False, on_epoch=True, on_step=False)
        return loss
    

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        features, labels = batch # collect input
        predictions = self.forward(features) # run input through the neural network
        loss, log_params = self.hparams.val_loss_f(predictions, labels)
        log_params = {f'test_{key}': value for key, value in log_params.items()}
        self.log_dict(log_params, prog_bar=False, on_epoch=True, on_step=False, add_dataloader_idx=False)
        return loss
    

    def predict_step(self, batch, batch_idx):
        self.eval()  # set the model to evaluation mode
        with torch.no_grad():  # disable gradient computation
            features, labels = batch
            predictions = self.forward(features)
        return predictions, labels



class ReconstructionDataModuleFolds(pl.LightningDataModule):
    def __init__(self, 
                 data_dir: str = data_path, 
                 exclusion_list: list[str] = [],
                 batch_size: int = 32, 
                 start_year: int = 1901,
                 end_year: int = 2020,
                 features: list[str] = [],
                 targets: list[str] = [],
                 static_features: list[str] = [],
                 spinup_length: int = 24,
                 prediction_length = 12,
                 feature_transform: list = [], # a list of tuples e.g., [(func1, [var1, var2]), (func2, ['all']), etc.]
                 target_transform: list = [],
                 folds: list[str] = [],
                 val_folds: list[list[str]] = [],
                 test_folds: list[list[str]] = [],
                 init_normalize = True,
                 init_normalize_targets = False,
                 weighted_sampling = False,
                 pred = False,
                 dataloader_kwargs = {},
                 num_workers: int = 8,
                 seed: int = SEED):
        super().__init__()
        self.save_hyperparameters()
        self.train_x_mean = 0
        self.train_x_std = 1
        self.train_y_mean = 0
        self.train_y_std = 1
        self.val_folds_idx = []
        self.test_folds_idx = []
        self.val_folds = []
        self.test_folds = []
        if not features:
            raise KeyError('No features given, must specify at least one feature!')
        if not targets:
            raise KeyError('No targets given, must specify at least one target!')

        self.create_dataset(exclusion_list=exclusion_list)
        self.check_vars_in_ds(variables=features+targets+static_features)
        self.check_regions()
        if self.hparams.pred:
            self.nonnan_targets = ~np.isnan(self.ds[self.hparams.targets].isel(time=(self.ds[self.hparams.targets].time>=np.datetime64(datetime.datetime(1997, 1, 1, 0, 0, 0)))).to_array()).sum(dim=['variable', 'time']).values.astype(bool) 
        else:
            self.nonnan_targets = ~np.isnan(self.ds[self.hparams.targets].isel(time=(self.ds[self.hparams.targets].time>=np.datetime64(datetime.datetime(self.hparams.start_year, 1, 1, 0, 0, 0)))).to_array()).sum(dim=['variable', 'time']).values.astype(bool) 

        self.folds_idcs = [[AR6_name_idx[regionname] for regionname in fold] for fold in folds]
        self.folds_coords = [self.get_masked_coords(fold_idcs) for fold_idcs in self.folds_idcs]
        self.all_coords = [coord for coords in self.folds_coords for coord in coords]
        self.val_coords = []
        self.test_coords = []

        if 'pin_memory' not in dataloader_kwargs:
            self.hparams.dataloader_kwargs['pin_memory'] = True
        
        if 'persistent_workers' not in dataloader_kwargs:
            self.hparams.dataloader_kwargs['persistent_workers'] = True

        self.split_folds(val_folds=val_folds, test_folds=test_folds)
        self.preprocess_data()       
        self.generate_datasets(normalize=init_normalize, normalize_targets=init_normalize_targets)

        
    def create_dataset(self, exclusion_list):
        start_year_spinup = math.floor(self.hparams.start_year - self.hparams.spinup_length/12)
        start_month_spinup = (1-self.hparams.spinup_length)%12
        if start_month_spinup == 0:
            start_month_spinup = 12
        file_list = [os.path.join(self.hparams.data_dir, filename) for filename in os.listdir(self.hparams.data_dir) if (os.path.isfile(os.path.join(self.hparams.data_dir, filename)) and filename not in exclusion_list)]
        self.ds = xr.open_mfdataset(file_list, preprocess=partial(preprocess, variables=self.hparams.features+self.hparams.targets+self.hparams.static_features, start_year=start_year_spinup, start_month=start_month_spinup, end_year=self.hparams.end_year), engine='netcdf4', combine='nested', chunks={})
        self.ds = self.ds.chunk(chunks={'time': 'auto', 'lat': 'auto', 'lon': 'auto'})
        if self.hparams.pred:
            if start_year_spinup < 1901:
                stitching_years = 1901-start_year_spinup
                # Stitch the first X months to the front so the spinup won't disregard them.
                first_n = self.ds.isel(time=slice(0, self.hparams.spinup_length))
                first_n['time'] = first_n['time'].values.astype('datetime64[M]') - stitching_years*12
                self.ds = xr.concat([first_n, self.ds], dim='time', join='exact')
            self.train_start_idx = int(np.where(self.ds.time.dt.year == 1997)[0][0])
            self.train_stop_idx = int(np.where(self.ds.time.dt.year == 2019)[0][-1])+1
        regionnames = np.full((360, 720), np.nan)
        for regionname, region in AR6_masks.items():
            regionnames[~region.data] = regionname
        self.ds = self.ds.assign_coords(coords={'Region': (('lat', 'lon'), regionnames)}).chunk(chunks={'time': -1, 'lat': 'auto', 'lon': 'auto'})
        self.ds.attrs['region_encoding'] = str(AR6_idx_name)
    
    def check_vars_in_ds(self, variables: str | list[str]) -> None:
        variables = [variables] if isinstance(variables, str) else variables
        missing_vars = [var for var in variables if var not in self.ds.data_vars]
        if missing_vars:
            raise KeyError(
                f'variable(s) `{"`, `".join(missing_vars)}` not found in the provided dataset `ds`. '
                f'Valid variables are: `{"`, `".join(list(self.ds.data_vars))}`.'
            )
            
    def check_regions(self):
        regions = [regionname for fold in self.hparams.folds for regionname in fold]
        wrong_regions = [region for region in regions if region not in AR6_name_idx.keys()]
        if wrong_regions:
            raise KeyError(
                f'region(s) `{"`, `".join(wrong_regions)}` do not exist!'
            )

        if self.hparams.val_folds:
            if not all([set(val_fold) in [set(fold) for fold in self.hparams.folds] for val_fold in self.hparams.val_folds]):
                raise KeyError(
                    f'val_fold(s) `{"`, `".join(self.hparams.val_folds)}` not in folds!'
                )
            
        if self.hparams.test_folds:
            if not all([set(test_fold) in [set(fold) for fold in self.hparams.folds] for val_fold in self.hparams.test_folds]):
                raise KeyError(
                    f'test_fold(s) `{"`, `".join(self.hparams.test_folds)}` not in folds!'
                )

    def get_masked_coords(self, region_idcs):
        mask = self.ds.Region.isin(region_idcs)
        mask = mask & self.nonnan_targets
        masked_indices = mask.data.nonzero()
        masked_lats = mask.lat.isel(lat=masked_indices[0])
        masked_lons = mask.lon.isel(lon=masked_indices[1])
        masked_coords = list(zip(masked_lats.values, masked_lons.values))
        return masked_coords
    
    def normalize_features(self):
        # Initialize accumulators
        n_total = 0
        mean_acc = None
        m2_acc = None  # Accumulator for the sum of squares of differences from the mean

        for obj in [ds for idx, ds in enumerate(self.fold_datasets) if idx not in self.val_folds_idx + self.test_folds_idx]:
            x = obj.x
            if self.hparams.pred:
                x = x[:, self.train_start_idx:self.train_stop_idx]
            n = x.shape[0] * x.shape[1]  # Number of samples in the current tensor
            mean = x.mean(dim=(0, 1))  # Mean over 'sample' and 'time' for each variable
            variance = x.var(dim=(0, 1), unbiased=False)  # Variance over 'sample' and 'time'
            #max = x.max(dim=(0, 1)) # Max over 'sample' and 'time'

            if mean_acc is None:
                # Initialize accumulators with the first DataArray
                mean_acc = mean
                m2_acc = variance * n
                n_total = n
            else:
                # Update accumulators incrementally
                delta = mean - mean_acc
                n_total += n
                mean_acc = mean_acc + delta * (n / n_total)
                m2_acc = m2_acc + variance * n + (delta**2) * (n_total - n) * n / n_total

        # Finalize mean and standard deviation
        self.train_x_mean = mean_acc
        #self.train_x_mean[self.hparams.static_features] = 0
        self.train_x_std = torch.sqrt(m2_acc / n_total)
        self.train_x_std = torch.max(self.train_x_std, torch.full_like(self.train_x_std, 1e-3)) # Floor stddev to 1e-3 for low variance features
        #self.train_x_std[self.hparams.static_features] = max[self.hparams.static_features]

        for ds in self.fold_datasets:
            ds.normalize(mean=self.train_x_mean, std=self.train_x_std)
        return None
    
    def unnormalize_features(self):
        for ds in self.fold_datasets:
            ds.unnormalize(mean=self.train_x_mean, std=self.train_x_std)
        self.train_x_mean = 0
        self.train_x_std = 1
        return None
    
    def normalize_targets(self):
        # Initialize accumulators
        n_total = 0
        mean_acc = None
        m2_acc = None  # Accumulator for the sum of squares of differences from the mean

        for obj in [ds for idx, ds in enumerate(self.fold_datasets) if idx not in self.test_folds_idx]:
            y = obj.y
            if self.hparams.pred:
                y = y[:, self.train_start_idx:self.train_stop_idx]
            else:
                y = y[:, self.hparams.spinup_length:]
            n = y.shape[0] * y.shape[1]  # Number of samples in the current tensor
            mean = y.nanmean(dim=(0, 1))  # Mean over 'sample' and 'time' for each variable
            variance = y.var(dim=(0, 1), unbiased=False)  # Variance over 'sample' and 'time'

            if mean_acc is None:
                # Initialize accumulators with the first DataArray
                mean_acc = mean
                m2_acc = variance * n
                n_total = n
            else:
                # Update accumulators incrementally
                delta = mean - mean_acc
                n_total += n
                mean_acc = mean_acc + delta * (n / n_total)
                m2_acc = m2_acc + variance * n + (delta**2) * (n_total - n) * n / n_total

        # Finalize mean and standard deviation
        self.train_y_mean = mean_acc
        self.train_y_std = torch.sqrt(m2_acc / n_total)

        for ds in self.fold_datasets:
            ds.normalize_targets(mean=self.train_y_mean, std=self.train_y_std)
        return None
    
    def unnormalize_targets(self):
        for ds in self.fold_datasets:
            ds.unnormalize_targets(mean=self.train_y_mean, std=self.train_y_std)
        self.train_y_mean = 0
        self.train_y_std = 1
        return None

    def preprocess_data(self):
        self.ds_stacked = self.ds.stack(sample=['lat', 'lon'])
        self.x = self.ds_stacked[self.hparams.features].chunk({'time': -1, 'sample': 'auto'}).sel(sample=self.all_coords).transpose('sample', 'time').to_array()
        self.y = self.ds_stacked[self.hparams.targets].chunk({'time': -1, 'sample': 'auto'}).sel(sample=self.all_coords).transpose('sample', 'time').to_array()

        if self.hparams.feature_transform:
            for (func, variables) in self.hparams.feature_transform:
                if variables == 'all' or variables == ['all']:
                    boolean_var_list = [True for var in self.x['variable']]
                else:
                    if isinstance(variables, str):
                        variables = [variables]
                    boolean_var_list = [var in variables for var in self.x['variable']]
                self.x.loc[boolean_var_list] = func(self.x.loc[boolean_var_list])
            #self.x = self.hparams.feature_transform(self.x)
        if self.hparams.target_transform:
            self.y = self.hparams.target_transform(self.y)
            
        self.ds_transformed = xr.concat([self.x, self.y], dim='variable')
        #self.ds_transformed = self.ds_transformed.load()
        
    def generate_datasets(self, normalize=True, normalize_targets=True):
        self.fold_datasets = [self.ds_transformed.sel(sample=fold_coords) for fold_coords in self.folds_coords]
        self.fold_datasets = [ReconstructionDataset(dataset, features=self.hparams.features, targets=self.hparams.targets, spinup_length=self.hparams.spinup_length, prediction_length=self.hparams.prediction_length) for dataset in self.fold_datasets]
        self.change_val_test_folds(val_folds=self.hparams.val_folds, test_folds=self.hparams.test_folds, normalize=normalize, normalize_targets=normalize_targets)


    def generate_dataloaders(self):
        train_set = MultipleDataset([dataset for idx, dataset in enumerate(self.fold_datasets) if idx not in self.val_folds_idx+self.test_folds_idx])
        val_set = MultipleDataset([dataset for idx, dataset in enumerate(self.fold_datasets) if idx in self.val_folds_idx])
        test_set = MultipleDataset([dataset for idx, dataset in enumerate(self.fold_datasets) if idx in self.test_folds_idx])
        #pred_set = MultipleDataset(self.fold_datasets)

        if self.hparams.weighted_sampling:
            train_sampler = WeightedRandomSampler(weights=train_set.weights, num_samples=len(train_set), replacement=True)
            self.train_loader = DataLoader(train_set, sampler=train_sampler, batch_size=self.hparams.batch_size, shuffle=True, num_workers=self.hparams.num_workers, **self.hparams.dataloader_kwargs)
        else:
            self.train_loader = DataLoader(train_set, batch_size=self.hparams.batch_size, shuffle=True, num_workers=self.hparams.num_workers, **self.hparams.dataloader_kwargs)
        
        self.val_loader = DataLoader(val_set, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_workers, **self.hparams.dataloader_kwargs)
        self.test_loader = DataLoader(test_set, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_workers, **self.hparams.dataloader_kwargs)
        #self.pred_loader = DataLoader(pred_set, batch_size=self.hparams.batch_size, shuffle=False, num_workers=min(4, self.hparams.num_workers), **self.hparams.dataloader_kwargs)

        
    def get_dataloader(self, mode):        
        if mode.lower() == 'train':
            return self.train_loader
        elif mode.lower() == 'val' or mode.lower() == 'validation':
            return self.val_loader
        elif mode.lower() == 'test':
            return self.test_loader
        #elif mode.lower() == 'pred' or mode.lower() == 'predict':
            #return self.pred_loader
        raise ValueError('Wrong value for mode')
    
    def get_target_zero(self):
        return ((torch.Tensor([0])-self.train_y_mean) / self.train_y_std).item()


    def reconstruct_pedictions(self, predictions, mode):
        if mode.lower() == 'train':
            coords = self.train_coords
        elif mode.lower() == 'val' or mode.lower() == 'validation':
            coords = self.val_coords
        elif mode.lower() == 'test':
            coords = self.test_coords
        #elif mode.lower() == 'pred' or mode.lower() == 'predict':
            #coords = self.all_coords
        raise ValueError('Wrong value for mode')
        assert len(coords) == predictions.shape[0], "Shape of predictions does not match the 'mode' dataset, wrong mode?"


    def change_prediction_start(self, new_prediction_start_idx):
        for ds in self.fold_datasets:
            ds.change_prediction_start(new_prediction_start_idx)
        self.generate_dataloaders()

    
    def change_prediction_length(self, new_pred_length):
        self.hparams.prediction_length = new_pred_length
        for ds in self.fold_datasets:
            ds.change_prediction_length(new_pred_length)
        self.generate_dataloaders()


    def change_val_test_folds(self, val_folds=None, test_folds=None, normalize=True, normalize_targets=True, pred_length=0):
        self.split_folds(val_folds=val_folds, test_folds=test_folds, pred_length=pred_length)
        if normalize:
            if isinstance(self.train_x_mean, torch.Tensor):
                self.unnormalize_features()
            self.normalize_features()
        if normalize_targets:
            if isinstance(self.train_y_mean, torch.Tensor):
                self.unnormalize_targets()
                print('unnormalized')
            self.normalize_targets() 
            print('normalized')
        self.generate_dataloaders()
  
    def split_folds(self, val_folds=None, test_folds=None, pred_length=0):
        if val_folds:
            self.val_folds_idx = []
            self.hparams.val_coords = []
            if self.hparams.val_folds:
                for val_fold in self.hparams.val_folds:
                    val_fold_idx = self.hparams.folds.index(val_fold)
                    self.fold_datasets[val_fold_idx].change_prediction_length(self.hparams.prediction_length)
            for val_fold in val_folds:
                val_fold_idx = self.hparams.folds.index(val_fold)
                self.val_folds_idx.append(val_fold_idx)
                self.hparams.val_coords += self.folds_coords[val_fold_idx]
                self.fold_datasets[val_fold_idx].change_prediction_length(pred_length)
            self.hparams.val_folds = val_folds
            self.val_coords = [fold_coord for idx, fold_coord in enumerate(self.folds_coords) if idx in self.val_folds_idx]
            self.val_coords = [coord for coords in self.val_coords for coord in coords]
            self.val_folds = val_folds

        if test_folds:
            self.test_folds_idx = []
            self.hparams.test_coords = []
            if self.hparams.test_folds:
                for test_fold in self.hparams.test_folds:
                    test_fold_idx = self.hparams.folds.index(test_fold)
                    if test_fold_idx not in self.val_folds_idx: # Only set predicition length to default if the old test fold is not (one of) the new val fold(s)
                        self.fold_datasets[test_fold_idx].change_prediction_length(self.hparams.prediction_length)
            for test_fold in test_folds:
                test_fold_idx = self.hparams.folds.index(test_fold)
                self.test_folds_idx.append(test_fold_idx)
                self.hparams.test_coords += self.folds_coords[test_fold_idx]
                self.fold_datasets[test_fold_idx].change_prediction_length(pred_length)
            self.hparams.test_folds = test_folds
            self.test_coords = [fold_coord for idx, fold_coord in enumerate(self.folds_coords) if idx in self.test_folds_idx]
            self.test_coords = [coord for coords in self.test_coords for coord in coords]
            self.test_folds = test_folds
        
        self.train_coords = [fold_coord for idx, fold_coord in enumerate(self.folds_coords) if idx not in self.val_folds_idx + self.test_folds_idx]
        self.train_coords = [coord for coords in self.train_coords for coord in coords]



class NLLPredWrapper(nn.Module):
    def __init__(self, base_model, train_y_mean, train_y_std, use_last_timestep=True):
        super().__init__()
        self.base_model = base_model
        # store as parameters/buffers so they move with .to(device)
        self.register_buffer("train_y_mean", torch.as_tensor(train_y_mean))
        self.register_buffer("train_y_std", torch.as_tensor(train_y_std))
        self.use_last_timestep = use_last_timestep

    def forward(self, x):
        """
        x: (batch, seq_in, features)
        returns: (batch,) final decoded prediction in real space
        """
        preds = self.base_model(x)          # (batch, T, 3)

        if self.use_last_timestep:
            preds = preds[:, -1, :]         # (batch, 3)

        pred_class = preds[..., 0:1]        # (batch, 1)
        pred_reg   = preds[..., 1:2]        # normalized log-mean
        pred_logvar = preds[..., 2:3]       # normalized log-var

        # denormalize μ_log and σ_log^2
        mu_log = pred_reg * self.train_y_std + self.train_y_mean
        var_log = torch.exp(pred_logvar) * (self.train_y_std ** 2)

        # lognormal expectation: E[y] = exp(μ + 0.5 σ^2)
        logmean_real = mu_log + 0.5 * var_log
        y_hat = torch.exp(logmean_real)     # (batch, 1)

        y_hat_masked = ((pred_class>0).float() * y_hat).squeeze(-1)     # (batch)
        max_tensor = torch.full_like(y_hat_masked, 100)

        return torch.min(max_tensor, y_hat_masked)         # (batch,)
            