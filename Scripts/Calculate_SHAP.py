import os
import random
import glob
from pathlib import Path
os.chdir(Path('/vscmnt/brussel_pixiu_data/_data_brussel/vo/000/bvo00012/vsc10262/Wildfires/WP1_Long_Term_BA_Reconstruction/Scripts'))
import torch
import optuna
import dask
import pickle
import shap
from tqdm import tqdm
import numpy as np

from functions import *
from global_variables import *
from pl_classes import *

torch.set_float32_matmul_precision('medium')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
dask.config.set(scheduler='synchronous')
torch.multiprocessing.set_start_method("spawn", force=True)


n_samples = 50
pred_start_idx = 120
pred_length = 60

if __name__ == "__main__":
    run_name = 'mse_1_64_log_None_ReLU_5_grasses_new_ordering' #'default' #'imbalanced' #'2024-10-02_154002' #False
    if not run_name:
        logs_path = max([el for el in glob.glob(os.path.join(model_path, 'Logs', '*')) if not el.endswith('lightning_logs')] , key=os.path.getctime)
        study_name = os.path.basename(logs_path)
    else:
        logs_path = os.path.join(model_path, 'Logs', run_name)
    folds = [dirname[5:].split('_') for dirname in os.listdir(logs_path) if (os.path.isdir(os.path.join(logs_path, dirname)) and dirname != 'lightning_logs')]

    dm_filename = Path(r'/vscmnt/brussel_pixiu_data/_data_brussel/vo/000/bvo00012/vsc10262/Wildfires/WP1_Long_Term_BA_Reconstruction/Supplementary_Data/log_reduced_dm_grasses.pkl')
    with open(dm_filename, "rb") as f:
        dm = pickle.load(f)

    dm.hparams.num_workers = 2
    dm.hparams['dataloader_kwargs'] = {}
    dm.change_prediction_length(pred_length)

    sample_index = dm.ds_transformed.sample.to_index()  # MultiIndex with levels (lat, lon)

    # Prepare a dictionary to hold the selected points for each boolean variable.
    selected_points = {idx: {} for idx in range(5)}

    # Get the coordinate arrays from ds.
    lat_vals = gfed_mask["lat"].values   # 1D array of latitudes
    lon_vals = gfed_mask["lon"].values   # 1D array of longitudes

    # Create 2D grids of coordinates (make sure you use the same indexing as your dataset)
    lat_grid, lon_grid = np.meshgrid(lat_vals, lon_vals, indexing="ij")  # shapes: (nlat, nlon)

    # Loop through each variable in ds.
    for regionname in gfed_mask.data_vars:
        # We'll work only on boolean variables.
        if gfed_mask[regionname].dtype != bool or regionname == 'OCEAN':
            continue

        # Get the boolean mask for this variable (shape: (nlat, nlon))
        mask = gfed_mask[regionname].values

        # Get all (lat, lon) points where the variable is True.
        # Flatten the grids and mask:
        lat_flat = lat_grid.flatten()
        lon_flat = lon_grid.flatten()
        mask_flat = ~mask.flatten()
        
        # Create a list of (lat, lon) tuples for the True cells.
        points = [(lat, lon) for lat, lon, valid in zip(lat_flat, lon_flat, mask_flat) if valid]
        
        # Now filter to only those points that are in dm.ds_transformed.sample.
        common_points = [pt for pt in points if pt in sample_index]
        
        if not common_points:
            print(f"No common points for '{regionname}' found in dm.ds_transformed.sample")
            continue
        for idx in range(5):
            # Randomly select up to n_samples points.
            n_select = min(n_samples, len(common_points))
            selected = random.sample(common_points, n_select)
        
            # Store the selected points
            selected_points[idx][regionname] = selected 

    explanations = {idx: {region: [] for region in selected_points[idx].keys()} for idx in range(5)}

    test_fold_paths = [os.path.join(logs_path, directory_name) for directory_name in os.listdir(logs_path) if directory_name.startswith('test_')]    
    for test_fold_idx, test_fold_path in enumerate(test_fold_paths):
        test_name = os.path.basename(os.path.normpath(test_fold_path))
        test_regions = test_name.split('_')[1:]
        test_fold_path_studies = [os.path.join(test_fold_path, study_name) for study_name in os.listdir(test_fold_path)]
        test_fold_path_studies.sort(key=lambda x: os.path.getmtime(x))            
        print(f'Starting test {test_regions}')      
    
        for study_idx, study_path in enumerate(test_fold_path_studies):
            study_name = study_path.split('/')[-1]
            val_regions = study_name.split('.')[0].split('_')[1:]
            for idx, item in enumerate(val_regions):
                if '-' in item:
                    idx_item = val_regions[idx].split('-')
                    val_regions = [val_regions[:idx] + [idx_item[0]], [idx_item[1]] + val_regions[idx+1:]]
            dm.change_val_test_folds(val_folds=val_regions, test_folds=[test_regions])
            optuna_study = optuna.load_study(study_name=study_name, storage=f"sqlite:///{study_path}/optuna_study.db") # storage=f"sqlite:///{study_path}"
            version_number = optuna_study.best_trial.number #+ 5*study_idx + num_runs*5*test_fold_idx # Change here (remove + 5*study_idx + num_runs*5*test_fold_idx)
            selected_model = f'version_{version_number}'
            model_version = os.path.join(study_path, 'lightning_logs', selected_model) # Change logs_path to study_path
            version_list = [pathname for pathname in glob.glob(os.path.join(model_version, 'checkpoints', '*')) if pathname.split('/')[-1].startswith('epoch')]
            last_version = max(version_list, key=os.path.getctime)
            best_version = torch.load(last_version, map_location="cpu")['callbacks']["ModelCheckpoint{'monitor': 'val_loss', 'mode': 'min', 'every_n_train_steps': 0, 'every_n_epochs': 1, 'train_time_interval': None}"]['best_model_path'].split('/')[-1]
            best_version = [filename for filename in version_list if best_version in filename][0]
            model = LightningLSTM.load_from_checkpoint(best_version)#.to(device)
            model.eval()
            print(f'Starting val {val_regions}')

            background_X = []
            for idx, batch in enumerate(dm.get_dataloader(mode='train')):
                # Assuming each batch is a tuple (X, y) or (X,) depending on your dataset
                X = batch[0]  # adjust this if your dataset returns a dict or something else
                background_X.append(X)  # move to CPU and convert to numpy
                if idx == 3: # Use 4 batches as background
                    break
            # Concatenate along the first dimension (batch dimension)
            background_X = torch.concatenate(background_X, axis=0)
            print('Background shape: ', background_X.shape)

            for idx, (region, coords) in tqdm(enumerate(selected_points[study_idx].items())):
                if region != 'CEAM':
                    continue
                test_coords = [coord for coord in coords if coord in dm.test_coords]
                if test_coords:
                    region_data_arr = dm.ds_transformed.sel(sample=test_coords)
                    custom_set = ReconstructionDataset(region_data_arr, features=dm.hparams.features, targets=dm.hparams.targets, spinup_length=dm.hparams.spinup_length, prediction_length=0)
                    custom_set.normalize(mean=dm.train_x_mean, std=dm.train_x_std)
                    dl = DataLoader(custom_set, batch_size=len(test_coords), shuffle=False, num_workers=1)
                    X = next(iter(dl))[0][:, pred_start_idx:pred_start_idx+pred_length+36, :]
                    explainer = shap.GradientExplainer(model, background_X)
                    explanations[study_idx][region].append(explainer(X))

    with open(os.path.join(logs_path, 'explanations_CEAM.pickle'), 'wb') as handle:
        pickle.dump(explanations, handle, protocol=pickle.HIGHEST_PROTOCOL)