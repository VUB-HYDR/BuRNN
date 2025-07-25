import os
import glob
from pathlib import Path
os.chdir(Path('/vscmnt/brussel_pixiu_data/_data_brussel/vo/000/bvo00012/vsc10262/Wildfires/WP1_Long_Term_BA_Reconstruction/Scripts'))
from functools import partial
import datetime
import fnmatch
import warnings
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import lstsq
from scipy import stats
import xarray as xr
import cf_units
import iris
import torch
import torchvision.transforms as transforms
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.tuner import Tuner
from pytorch_lightning.callbacks import ModelCheckpoint
import optuna
import random
import dask
import pickle

from functions import *
from global_variables import *
from pl_classes import *

torch.set_float32_matmul_precision('medium')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
dask.config.set(scheduler='synchronous')
torch.multiprocessing.set_start_method("spawn", force=True)

if __name__ == "__main__":
    exclusion_list = ['FireCCILT11.nc', 'ssib4-triffid-fire_pfts.nc', 'spei_results.nc', 'fwi_month.nc', 'states_1901_2019_monthly_remapped_corrected.nc', 'spei04.nc', 'classic_pfts_monthly.nc']
    features = [
        #'primf', 'primn', 'secdf', 'secdn', 'urban', 'c3ann', 'c4ann', 'c3per', 'c4per', 'c3nfx', 'pastr', 'range', 'secmb', 'secma', #14
        'managed_pastures', 'rangeland', #2
        'lightning-density', #1
        'rural-population', 'urban-population', #2
        'gdp', #1
        'tasmax_monmean', 'pr_monmean', 'sfcwind_monmean', #3
        'fwi', #1
        'spei01', 'spei03', 'spei06', #3
        'gpp-total', 'cveg-total', 'lai-total', #3
        #'Broadleaf evergreen tree', 'Needleleaf evergreen tree', 'grass', 'savanna', 'shrub', 'tundra shrub', 'Broadleaf deciduous tree'
        'PCT_URBAN', 'PCT_LAKE', 'PCT_CROP', 'Bare Ground', 'Needleleaf tree', 'Broadleaf evergreen tree', 'Broadleaf deciduous tree', 'Broadleaf shrub - temperate', 'Broadleaf deciduous shrub - boreal', 'C3 grass', 'C4 grass' #11
        , 'C4 grass Americas', 'C4 grass Africa', 'C4 grass Euraustralasia'
    ]
    targets = ['percentage_of_area_burned']

    n_trials = 3

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


    regions = 'all'
    val_regions = []#['CAF', 'NSA', 'SAH', 'CAU', 'WSB', 'NWN', 'NEU']
    test_regions = []#['WSAF', 'SAM', 'NZ', 'SAU', 'RAR', 'CNA', 'SAS']
    feature_transforms = [(log_transform, ['lightning-density', 'rural-population', 'urban-population', 'gdp'])]#None#[normalize_features]
    target_transforms = [log_transform]
    '''
    if feature_transforms:
        feature_transform = transforms.Compose(feature_transforms)
    else:
        feature_transform = []
    '''
    feature_transform = feature_transforms
    if target_transforms:
        target_transform = transforms.Compose(target_transforms)
    else:
        target_transform = []
    num_workers = 7
    spinup_length = 36
    prediction_length = 36
    lambda_factor = 1
    batch_size = 32
    ba_as_predictor = False
    linear = True

    ###
    # Define name for the run here. Put to False for auto-generated name
    custom_name = f'mse_1_64_log_None_ReLU_5_grasses_new_ordering'
    ###

    pos_weight = torch.tensor(1, dtype=torch.float32) #6.2799
    bce_loss_fn = nn.BCEWithLogitsLoss(reduction='sum', pos_weight=pos_weight)


    def objective(trial, log_path):
        num_layers = 1# trial.suggest_int('num_layers', 1, 2)
        hidden_size = 64# trial.suggest_categorical('hidden_size', [32, 64, 128])
        activ_f1 = None
        activ_f2 = 'ReLU'
        dropout = 0.2
        max_epochs = 100
        learning_rate = 0.00075 #trial.suggest_float("lr", 1e-5, 1e-3, log=True) #0.001
        seed = trial.suggest_int('seed', 1, 9e7)
        #loss_f, dynamic_loss, num_targets = partial(Binary_and_MSELoss, bce_loss_fn=bce_loss_fn, lambda_factor=lambda_factor, alpha=1), True, len(targets)*2

        loss_f, dynamic_loss, num_targets = partial(MSELoss, lambda_factor=lambda_factor), False, len(targets)
        model = LightningLSTM(input_size=len(features), hidden_size=hidden_size, num_layers=num_layers, lstm_activ_f=activ_f1, linear_activ_f=activ_f2, output_size=num_targets, dropout=dropout, loss_f=loss_f, linear=linear, dynamic_loss=dynamic_loss, lr=learning_rate, spinup_length=spinup_length, ba_as_predictor=ba_as_predictor, seed=seed)

        early_stopping = EarlyStopping(
                monitor='val_loss',  # Metric to monitor
                patience=4,          # Number of epochs with no improvement after which training will be stopped
                verbose=True,
                mode='min'           # The mode can be 'min' for minimizing or 'max' for maximizing the monitored quantity
                )
                
        # Learning rate monitor
        lr_monitor = LearningRateMonitor(logging_interval='epoch')

        checkpoint_callback = ModelCheckpoint(save_top_k=5, monitor="val_loss", mode="min", save_last=True)
        # Create a logger
        tb_logger = pl_loggers.TensorBoardLogger(log_path)
        # Trainer with early stopping and learning ra   te monitor callbacks with logging
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            callbacks=[early_stopping, lr_monitor, checkpoint_callback],
            logger=tb_logger,
            enable_progress_bar=False,
            gradient_clip_val=0.5,
            accelerator="cuda",
            profiler=None
            )

        trainer.fit(model=model, train_dataloaders=dm.get_dataloader(mode='train'), val_dataloaders=dm.get_dataloader(mode='val'))

        val_loss = trainer.early_stopping_callback.best_score.detach()
        return val_loss

    if custom_name == False:
        current_date = datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S')
        logs_path = os.path.join(model_path, 'Logs', current_date)
    else:
        logs_path = os.path.join(model_path, 'Logs', custom_name)
    try:
        os.mkdir(logs_path)
    except FileExistsError:
        pass

    dm_filename = Path(r'/vscmnt/brussel_pixiu_data/_data_brussel/vo/000/bvo00012/vsc10262/Wildfires/WP1_Long_Term_BA_Reconstruction/Supplementary_Data/log_reduced_dm_grasses_feature_transformed.pkl')
    if dm_filename.is_file():
        with open(dm_filename, "rb") as f:
            dm = pickle.load(f)
            dm.hparams['dataloader_kwargs'] = {'persistent_workers': True}
    else:
        dm = ReconstructionDataModuleFolds(
            data_dir = data_path,
            exclusion_list = exclusion_list,
            batch_size = batch_size, 
            start_year = 1997,
            end_year = 2019,
            features = features,
            targets = targets,
            feature_transform=feature_transform,
            target_transform=target_transform,
            static_features = [],
            spinup_length = spinup_length,
            prediction_length = prediction_length,
            folds=folds,
            num_workers=num_workers,
            seed = SEED)
        with open(dm_filename, "wb") as f:
            pickle.dump(dm, f, pickle.HIGHEST_PROTOCOL)

    for test_idx, test_fold in enumerate(folds[::-1]):
        train_val_folds = folds[:test_idx] + folds[test_idx+1:]
        train_val_folds = [[train_val_folds[i], train_val_folds[(i+1)%len(train_val_folds)]] for i in range(0, len(train_val_folds), 2)]
        test_path = os.path.join(logs_path, 'test_'+'_'.join(test_fold))
        try:
            os.mkdir(test_path)
        except FileExistsError:
            continue
        print(f"Starting test regions: {' '.join(test_fold)}")
        for val_idx, val_folds in enumerate(train_val_folds):
            dm.change_val_test_folds(val_folds = val_folds, test_folds = [test_fold])
            valname = f"val_{'-'.join(['_'.join(fold) for fold in val_folds])}"
            val_path = os.path.join(os.path.join(test_path, valname))
            try:
                os.mkdir(val_path)
            except FileExistsError:
                continue
            study = optuna.create_study(direction='minimize', study_name=valname, storage=f"sqlite:///{os.path.join(val_path, 'optuna_study.db')}")
            objective_f = partial(objective, log_path = val_path)
            torch.cuda.synchronize()
            study.optimize(objective_f, n_trials=n_trials)
            print(f"Study path: {os.path.join(test_path, valname)}")