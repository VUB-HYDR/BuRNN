import os
import glob
import re
from pathlib import Path
SCRIPTS_PATH = Path('/vscmnt/brussel_pixiu_data/_data_brussel/vo/000/bvo00012/vsc10262/Wildfires/WP1_Long_Term_BA_Reconstruction/Scripts')
os.chdir(SCRIPTS_PATH)
import datetime
import math
import functools
from functools import partial
import warnings
import random
import fnmatch
import pickle
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr, pearsonr, beta, gaussian_kde
import dask
import iris
import iris.plot as iplt
import cf_units
import cartopy
import cartopy.crs as ccrs
from scipy.ndimage import convolve
import torch
import torchvision.transforms as transforms
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
import optuna

from functions import *
from global_variables import *
from pl_classes import *

INVERSE_FUNCS = {divide_by_100: lambda x: x.multiply(100),
                    log_transform: log_inverse,
                    log10_transform: log10_inverse
                         }



torch.set_float32_matmul_precision('medium')

exclusion_list = ['FireCCILT11.nc', 'ssib4-triffid-fire_pfts.nc', 'spei_results.nc', 'fwi_month.nc', 'states_1901_2019_monthly_remapped_corrected.nc', 'spei04.nc', 'classic_pfts_monthly.nc']
features = [
    #'primf', 'primn', 'secdf', 'secdn', 'urban', 'c3ann', 'c4ann', 'c3per', 'c4per', 'c3nfx', 'pastr', 'range', 'secmb', 'secma', #14
    'managed_pastures', 'rangeland', #2
    'lightning-density', #1
    'rural-population', 'urban-population', #2
    'gdp', #1
    'tasmax_monmean', 'pr_monmean', 'sfcwind_monmean', #3
    'fwi', #1
    #'fwi_normalised',
    #'spei01', 'spei03', 'spei06', #3
    'gpp-total', 'cveg-total', 'lai-total', #3
    #'Broadleaf evergreen tree', 'Needleleaf evergreen tree', 'grass', 'savanna', 'shrub', 'tundra shrub', 'Broadleaf deciduous tree'
    'PCT_URBAN', 'PCT_LAKE', 'PCT_CROP', 'Bare Ground', 'Needleleaf tree', 'Broadleaf evergreen tree', 'Broadleaf deciduous tree', 'Broadleaf shrub - temperate', 'Broadleaf deciduous shrub - boreal', 'C3 grass', 'C4 grass' #11
    #, 'C4 grass Americas', 'C4 grass Africa', 'C4 grass Euraustralasia'
]

folds = [['NWN', 'WAF', 'SEA', 'SSA'],
 ['NWS', 'RAR', 'CAF', 'EAU'],
 ['NCA', 'SES', 'NEU', 'SAS'],
 ['CNA', 'SEAF', 'WSB', 'NZ'],
 ['SAM', 'SAH', 'EAS', 'SAU'],
 ['NEN', 'SWS', 'MED', 'ECA'],
 ['CAR', 'WCE', 'ESAF', 'RFE'],
 ['WNA', 'ARP', 'ESB', 'NAU'],
 ['ENA', 'NES', 'MDG', 'TIB'],
 ['SCA', 'WSAF', 'WCA', 'CAU'],
 ['NSA', 'NEAF', 'EEU']]


targets = ['percentage_of_area_burned']
regions = 'all'
val_regions = []#['CAF', 'NSA', 'SAH', 'CAU', 'WSB', 'NWN', 'NEU']
test_regions = []#['WSAF', 'SAM', 'NZ', 'SAU', 'RAR', 'CNA', 'SAS']
feature_transforms = [(log_transform, ['lightning-density', 'rural-population', 'urban-population', 'gdp'])]#None#[normalize_features]
target_transforms = [log_transform]
feature_transform = feature_transforms
'''
if feature_transforms:
    feature_transform = transforms.Compose(feature_transforms)
else:
    feature_transform = []
'''
if target_transforms:
    target_transform = transforms.Compose(target_transforms)
else:
    target_transform = []

num_workers = 1
lambda_factor = 5
spinup_length = 36
prediction_length = 0
max_epochs = 250
batch_size = 32 

run_name = 'sanity_check' #'default' #'imbalanced' #'2024-10-02_154002' #False
if not run_name:
    logs_path = max([el for el in glob.glob(os.path.join(model_path, 'Logs', '*')) if not el.endswith('lightning_logs')] , key=os.path.getctime)
    study_name = os.path.basename(logs_path)
else:
    logs_path = os.path.join(model_path, 'Logs', run_name)
folds = [dirname[5:].split('_') for dirname in os.listdir(logs_path) if (os.path.isdir(os.path.join(logs_path, dirname)) and dirname != 'lightning_logs')]


with dask.config.set(**{'array.slicing.split_large_chunks': True}):
    dm_filename = Path(r'/vscmnt/brussel_pixiu_data/_data_brussel/vo/000/bvo00012/vsc10262/Wildfires/WP1_Long_Term_BA_Reconstruction/Supplementary_Data/log_dm_nospei_nograsses_normalisedtarget_full.pkl')
    if dm_filename.is_file():
        with open(dm_filename, "rb") as f:
            dm = pickle.load(f)
            #dm.change_prediction_length(0)
    else:
        dm = ReconstructionDataModuleFolds(
            data_dir = data_path,
            exclusion_list = exclusion_list,
            batch_size = batch_size, 
            start_year = 1901,
            end_year = 2020,
            features = features,
            targets = targets,
            feature_transform = feature_transform,
            target_transform = target_transform,
            static_features = [],
            spinup_length = spinup_length,
            prediction_length = prediction_length,
            folds = folds,
            init_normalize = False,
            num_workers = num_workers,
            pred = True,
            seed = SEED)
        with open(dm_filename, "wb") as f:
            pickle.dump(dm, f, pickle.HIGHEST_PROTOCOL)

# Early stopping callback
early_stopping = EarlyStopping(
    monitor='val_loss',  # Metric to monitor
    patience=4,          # Number of epochs with no improvement after which training will be stopped
    verbose=True,
    mode='min'           # The mode can be 'min' for minimizing or 'max' for maximizing the monitored quantity
)
# Learning rate monitor
lr_monitor = LearningRateMonitor(logging_interval='step')

# Create a logger   
tb_logger = pl_loggers.TensorBoardLogger(os.path.join(model_path, 'Logs'))
# Trainer with early stopping and learning rate monitor callbacks with logging
trainer = pl.Trainer(
    max_epochs=max_epochs,
    callbacks=[early_stopping, lr_monitor],
    logger=tb_logger
)

if not os.path.exists(os.path.join(logs_path, 'predicted_full_new.nc')):   
    test_fold_paths = [os.path.join(logs_path, directory_name) for directory_name in os.listdir(logs_path) if directory_name.startswith('test_')]

    num_runs = [os.path.join(logs_path, dirname) for dirname in os.listdir(logs_path) if os.path.isdir(os.path.join(logs_path, dirname))]
    num_runs = len([dirname for dirname in os.listdir(num_runs[0]) if (os.path.isdir(os.path.join(num_runs[0], dirname)) or dirname.endswith('.db'))])
    
    predictions_ds = xr.full_like(dm.ds_stacked[dm.hparams.targets], np.nan)
    #predictions_ds = predictions_ds.sel(time=slice(f'{dm.hparams.start_year}-01-01', f'{dm.hparams.end_year-1}-12-31'))
    if dm.hparams.prediction_length:
        predictions_ds = predictions_ds.isel(time=slice(dm.hparams.spinup_length, dm.hparams.prediction_length))
    else:
        predictions_ds = predictions_ds.isel(time=slice(dm.hparams.spinup_length, len(predictions_ds.time)))
    predictions_ds = predictions_ds.expand_dims({'version': num_runs}).assign_coords({'version': np.arange(num_runs)})
    predictions_ds.load() # Needed, otherwise it complains that it can't overwrite different non-loaded dask chunks?


    
    predictions_ds_class = xr.full_like(dm.ds_stacked[dm.hparams.targets], np.nan)
    predictions_ds_reg = xr.full_like(dm.ds_stacked[dm.hparams.targets], np.nan)
    predictions_ds_log_var = xr.full_like(dm.ds_stacked[dm.hparams.targets], np.nan)
    if dm.hparams.prediction_length:
        predictions_ds_class = predictions_ds_class.isel(time=slice(dm.hparams.spinup_length, dm.hparams.prediction_length))
        predictions_ds_reg = predictions_ds_reg.isel(time=slice(dm.hparams.spinup_length, dm.hparams.prediction_length))
        predictions_ds_log_var = predictions_ds_log_var.isel(time=slice(dm.hparams.spinup_length, dm.hparams.prediction_length))

    else:
        predictions_ds_class = predictions_ds_class.isel(time=slice(dm.hparams.spinup_length, len(predictions_ds_class.time)))
        predictions_ds_reg = predictions_ds_reg.isel(time=slice(dm.hparams.spinup_length, len(predictions_ds_reg.time)))
        predictions_ds_log_var = predictions_ds_log_var.isel(time=slice(dm.hparams.spinup_length, len(predictions_ds_log_var.time)))
    predictions_ds_class = predictions_ds_class.expand_dims({'version': num_runs}).assign_coords({'version': np.arange(num_runs)})
    predictions_ds_reg = predictions_ds_reg.expand_dims({'version': num_runs}).assign_coords({'version': np.arange(num_runs)})
    predictions_ds_log_var = predictions_ds_log_var.expand_dims({'version': num_runs}).assign_coords({'version': np.arange(num_runs)})
    predictions_ds_class.load()
    predictions_ds_reg.load()
    predictions_ds_log_var.load()
    

    
    for test_fold_idx, test_fold_path in enumerate(test_fold_paths):
        test_name = os.path.basename(os.path.normpath(test_fold_path))
        test_regions = test_name.split('_')[1:]
        test_fold_path_studies = [os.path.join(test_fold_path, study_name) for study_name in os.listdir(test_fold_path)]
        test_fold_path_studies.sort(key=lambda x: os.path.getmtime(x))            
        dm.change_val_test_folds(test_folds=[test_regions])
        coords = dm.test_coords
        
        predicted_unstacked_list = []

        if len(test_fold_path_studies) < 5:
            continue
        
        for study_idx, study_path in enumerate(test_fold_path_studies):
            study_name = study_path.split('/')[-1]
            val_regions = study_name.split('.')[0].split('_')[1:]
            val_folds = [val_fold.split('_') for val_fold in study_name[4:].split('-')]
            dm.change_val_test_folds(val_folds = val_folds, test_folds = [test_regions], normalize_targets=not bool(study_idx))
            dl = dm.get_dataloader(mode='test')
            optuna_study = optuna.load_study(study_name=study_name, storage=f"sqlite:///{study_path}/optuna_study.db") # storage=f"sqlite:///{study_path}"
            version_number = optuna_study.best_trial.number #+ 5*study_idx + num_runs*5*test_fold_idx # Change here (remove + 5*study_idx + num_runs*5*test_fold_idx)
            selected_model = f'version_{version_number}'
            model_version = os.path.join(study_path, 'lightning_logs', selected_model) # Change logs_path to study_path
            version_list = [pathname for pathname in glob.glob(os.path.join(model_version, 'checkpoints', '*')) if pathname.split('/')[-1].startswith('epoch')]
            last_version = max(version_list, key=os.path.getctime)
            best_version = torch.load(last_version, map_location="cpu")['callbacks']["ModelCheckpoint{'monitor': 'val_loss', 'mode': 'min', 'every_n_train_steps': 0, 'every_n_epochs': 1, 'train_time_interval': None}"]['best_model_path'].split('/')[-1]
            best_version = [filename for filename in version_list if best_version in filename][0]
            model = LightningLSTM.load_from_checkpoint(best_version)
            model.eval()

            if model.hparams.input_size != len(features):
                raise ValueError(f'Wrong features in dataloader, expected {model.hparams.input_size} features but got {len(features)}')
    
            predictions_true_values = trainer.predict(model, dataloaders=dl)
            predictions = torch.cat([prediction for prediction, true_value in predictions_true_values])

            if model.hparams.val_loss_f == Binary_and_MSELoss or (isinstance(model.hparams.val_loss_f, partial) and model.hparams.val_loss_f.func == Binary_and_MSELoss):
                predictions_class = predictions[:, :, 0][:, :, None]
                predictions_reg = predictions[:, :, 1][:, :, None]
                predictions_reg = (predictions_reg * dm.train_y_std) + dm.train_y_mean
                for func in target_transforms[::-1]:
                    if isinstance(func, functools.partial):
                        func = func.func # TO DO: ALSO COPY THE ARGS OF FUNC
                predictions = (predictions_class>0).float() * log_inverse(predictions_reg)
                predictions = predictions.numpy()
                

                for func in target_transforms[::-1]:
                    if isinstance(func, functools.partial):
                        func = func.func # TO DO: ALSO COPY THE ARGS OF FUNC
                    predictions[predictions>0] = INVERSE_FUNCS[func](predictions[predictions>0])

            
            elif model.hparams.val_loss_f == Binary_and_NLLLoss or (isinstance(model.hparams.val_loss_f, partial) and model.hparams.val_loss_f.func == Binary_and_NLLLoss):
                predictions_class = predictions[:, :, 0][:, :, None]
                predictions_reg = predictions[:, :, 1][:, :, None]
                predictions_log_var = predictions[:, :, 2][:, :, None]
                predictions_reg = (predictions_reg * dm.train_y_std) + dm.train_y_mean
                predictions_log_var = torch.exp(predictions_log_var) * (dm.train_y_std ** 2)
                #predictions = (predictions_class>0).float() * log_inverse(predictions_reg + 0.5 * predictions_log_var)
                predictions = torch.sigmoid(predictions_class) * log_inverse(predictions_reg + 0.5 * predictions_log_var)
                predictions = predictions.numpy()

            else:
                for func in target_transforms[::-1]:
                    if isinstance(func, functools.partial):
                        func = func.func # TO DO: ALSO COPY THE ARGS OF FUNC
                    predictions = INVERSE_FUNCS[func](predictions)
            
            for target_idx, target in enumerate(targets):
                predictions_ds[dm.hparams.targets[0]].loc[study_idx, slice(None), coords] = predictions.squeeze().T#[:, :, target_idx]
                #predictions_ds_class[dm.hparams.targets[0]].loc[study_idx, slice(None), coords] = predictions_class.squeeze().T#[:, :, target_idx]
                predictions_ds_reg[dm.hparams.targets[0]].loc[study_idx, slice(None), coords] = predictions_reg.squeeze().T#[:, :, target_idx]
                predictions_ds_log_var[dm.hparams.targets[0]].loc[study_idx, slice(None), coords] = predictions_log_var.squeeze().T#[:, :, target_idx]

    
    predictions_ds.unstack().to_netcdf(os.path.join(logs_path, 'predicted_full_new.nc'))
    #predictions_ds_class.unstack().to_netcdf(os.path.join(logs_path, 'predicted_full_class.nc'))
    #predictions_ds_reg.unstack().to_netcdf(os.path.join(logs_path, 'predicted_full_reg.nc'))
    #predictions_ds_log_var.unstack().to_netcdf(os.path.join(logs_path, 'predicted_full_log_var.nc'))