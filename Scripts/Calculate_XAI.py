import os
import glob
from pathlib import Path
SCRIPTS_PATH = Path('/vscmnt/brussel_pixiu_data/_data_brussel/vo/000/bvo00012/vsc10262/Wildfires/WP1_Long_Term_BA_Reconstruction/Scripts')
os.chdir(SCRIPTS_PATH)
import pickle
from tqdm import tqdm

import seaborn as sns
import xarray as xr
import numpy as np
import pandas as pd
import dask
import optuna
from captum.attr import IntegratedGradients
import torch
from captum.attr import NoiseTunnel

from functions import *
from global_variables import *
from pl_classes import *


run_name = 'mse_1_64_log_NLL1_nospei_nograsses_seperateheads_normalizetargetsfold_lambda1000'
if not run_name:
    logs_path = max([el for el in glob.glob(os.path.join(model_path, 'Logs', '*')) if not el.endswith('lightning_logs')] , key=os.path.getctime)
    study_name = os.path.basename(logs_path)
else:
    logs_path = os.path.join(model_path, 'Logs', run_name)
folds = [dirname[5:].split('_') for dirname in os.listdir(logs_path) if (os.path.isdir(os.path.join(logs_path, dirname)) and dirname != 'lightning_logs')]

with dask.config.set(**{'array.slicing.split_large_chunks': True}):
    dm_filename = Path(r'/vscmnt/brussel_pixiu_data/_data_brussel/vo/000/bvo00012/vsc10262/Wildfires/WP1_Long_Term_BA_Reconstruction/Supplementary_Data/log_nospei_nograsses_normalizetargets.pkl')
    if dm_filename.is_file():
        with open(dm_filename, "rb") as f:
            dm = pickle.load(f)


gfed_mask_clean = gfed_mask.drop_vars("OCEAN")

coarse_da = xr.concat(
    [gfed_mask_clean[var] for var in gfed_mask_clean.data_vars],
    dim="coarse_region"
)
coarse_da = coarse_da.assign_coords(
    coarse_region=list(gfed_mask_clean.data_vars)
)

cell_area = calculate_area_weights(AR6_mask.isel(region=0).lat, AR6_mask.isel(region=0).lon)
cell_area = cell_area/cell_area.sum()

fine_ids = AR6_mask.region.values
coarse_ids = coarse_da.coarse_region.values

weights = xr.DataArray(
    np.zeros((len(coarse_ids), len(fine_ids))),
    coords={"coarse": coarse_ids, "fine": fine_ids},
    dims=("coarse", "fine")
)

for c in coarse_ids:
    coarse_mask_c = coarse_da.sel(coarse_region=c)

    for f in fine_ids:
        fine_mask_f = AR6_mask.sel(region=f)

        overlap = ~coarse_mask_c & ~fine_mask_f

        # area-weighted overlap fraction
        weights.loc[dict(coarse=c, fine=f)] = (cell_area * overlap).sum()
weights = weights / weights.sum(dim="fine")


device = "cuda" if torch.cuda.is_available() else "cpu"
g_region = gfed_mask_clean.to_array().argmin("variable")
g_names = list(gfed_mask_clean.keys())

# storage for attributions
region_attr_maps = {name: [] for name in g_names}   # 14 GFED regions
region_attr_maps['Global'] = []


gfed_X_means = xr.open_dataarray(os.path.join(supplementary_output_path, 'gfed_X_means.nc'))

test_fold_paths = [os.path.join(logs_path, directory_name) for directory_name in os.listdir(logs_path) if directory_name.startswith('test_')]
for test_fold_idx, test_fold_path in enumerate(test_fold_paths):

    test_name = os.path.basename(os.path.normpath(test_fold_path))
    test_regions = test_name.split('_')[1:]                 # list of region IDs in this test fold

    test_fold_path_studies = [os.path.join(test_fold_path, s) for s in os.listdir(test_fold_path)]
    test_fold_path_studies.sort(key=lambda x: os.path.getmtime(x))

    # adjust datamodule for this test fold
    dm.change_val_test_folds(test_folds=[test_regions])
    coords = dm.test_coords
    dl = dm.get_dataloader(mode='test')   # test dataloader for these regions

    # --------------------------------------------
    # Loop over 5 study paths per region fold
    # --------------------------------------------
    for study_idx, study_path in enumerate(test_fold_path_studies):

        study_name = study_path.split('/')[-1]
        val_regions = study_name.split('.')[0].split('_')[1:]

        # Load Optuna study → best trial
        optuna_study = optuna.load_study(
            study_name=study_name, 
            storage=f"sqlite:///{study_path}/optuna_study.db"
        )

        version_number = optuna_study.best_trial.number
        selected_model = f"version_{version_number}"
        model_version = os.path.join(study_path, "lightning_logs", selected_model)

        # all checkpoint files
        version_list = [
            p for p in glob.glob(os.path.join(model_version, "checkpoints", "*"))
            if p.split('/')[-1].startswith("epoch")
        ]
        last_version = max(version_list, key=os.path.getctime)
        # Get the *best* checkpoint path inside the Lightning checkpoint callback
        best_version = torch.load(last_version, map_location="cpu", weights_only=False)[
            'callbacks'
        ]["ModelCheckpoint{'monitor': 'val_loss', 'mode': 'min', 'every_n_train_steps': 0, 'every_n_epochs': 1, 'train_time_interval': None}"][
            'best_model_path'
        ].split('/')[-1]

        # match filename to actual checkpoint path
        best_version = [fn for fn in version_list if best_version in fn][0]

        # load the model
        model = LightningLSTM.load_from_checkpoint(best_version)
        model.to(device)
        model.eval()

        model = NLLPredWrapper(base_model=model, train_y_mean=dm.train_y_mean.to(device), train_y_std=dm.train_y_std.to(device), use_last_timestep=True)

        # --------------------------------------------------------
        # RUN CAPTUM INTEGRATED GRADIENTS FOR THIS MODEL
        # --------------------------------------------------------
        ig = IntegratedGradients(model)
        with captum_safe_mode(model):        
            batch_counter = 0
            # Loop through test batches
            for batch in tqdm(dl):
                X, y = batch
                X = X.to(device)
                y = y.to(device)
                batch_size = X.shape[0]

                # --------------------------------------------------------
                # Assign each sample to its GFED region
                # --------------------------------------------------------
                batch_region_coords = coords[batch_counter:batch_counter+batch_size]
                batch_counter += batch_size
                
                batch_region_ids = [g_names[g_region.sel(lat=lat, lon=lon).values.item()] for (lat, lon) in batch_region_coords]

                baseline = torch.from_numpy(gfed_X_means.loc[[gfed_name_idx[name] for name in batch_region_ids]].values.astype(np.float32)).to(device)[:, None, :]
                baseline = (baseline - dm.train_x_mean.to(device))/dm.train_x_std.to(device)
                # IG attribution
                attrs = []
                attrs_global = []
                for t in range(96, 144):
                    attr_x = X[:, t-36: t+1, :]
                    baseline_batch = baseline.repeat(1, 37, 1).requires_grad_(True)
                    baseline_batch = torch.zeros_like(baseline_batch)
                    attr = ig.attribute(attr_x, baselines=baseline_batch, target=None, n_steps=10)
                    attrs.append(np.abs(attr.detach().cpu()))

                    #baseline_global = torch.zeros_like(baseline_batch)
                    #attr_global = ig.attribute(attr_x, baselines=baseline_batch, target=None, n_steps=10)
                    #attrs_global.append(attr_global)
                
                attrs_mean = torch.stack(attrs).mean(dim=0)
                #attrs_global = torch.stack(attrs_global).mean(dim=0)

                for i, region_name in enumerate(batch_region_ids):
                    region_attr_maps[region_name].append(attrs_mean[i])
                    region_attr_maps['Global'].append(attrs_mean[i])

                




torch.save(region_attr_maps, os.path.join(logs_path, "IntegratedGradients_raw_zeros.pt"))
# First aggregate to a mean per region
region_mean_attrs = {
    r: np.mean(np.stack(arrs), axis=0)
    for r, arrs in region_attr_maps.items() if arrs
}

ds = xr.Dataset(
    {
        r: (("sample", "feature"), region_mean_attrs[r])
        for r in region_mean_attrs
    }
)

ds.to_netcdf(os.path.join(logs_path, 'IntegratedGradients.nc'))