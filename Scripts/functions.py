import math
import datetime
import numpy as np
import pandas as pd
import xarray as xr
import dask
import cftime
import cf_units
from iris import cube
from iris import coord_categorisation
from iris.time import PartialDateTime
import torch
import torch.nn as nn
from torchmetrics.functional import spearman_corrcoef, pearson_corrcoef
import warnings
from global_variables import *

def preprocess_time(ds):
    warnings.simplefilter(action="ignore", category=UserWarning) # Surpress some unimportant warnings
    ds = ds.convert_calendar('standard', align_on='date')
    ds['time'] = ds['time'].astype('datetime64[M]')
    warnings.simplefilter(action="default", category=UserWarning) 
    return ds


def create_n_folds(obj, n):
    k, m = divmod(len(obj), n)
    return list((obj[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n)))
    

def constrain_time(obj, start_year, end_year, start_month=1, end_month=12):
    # Constrains the time of an iris Cube or pandas DataFrame to be between the two dates (both included).
    if type(obj) == iris.cube.Cube:
        obj = add_year_coord(obj)
        start_time = PartialDateTime(year=start_year, month=start_month)
        end_time = PartialDateTime(year=end_year, month=end_month)
        return obj.extract(iris.Constraint(time=lambda cell: start_time <= cell.point <= end_time))
    
    elif type(obj) in (pd.core.frame.DataFrame, pd.core.series.Series):
        return obj.loc[f'{start_year}-{start_month}': f'{end_year}-{end_month}']
    
    elif type(obj) in (xr.Dataset, xr.DataArray):
        start_time = np.datetime64(datetime.datetime(start_year, start_month, 1, 0, 0, 0))
        if end_year:
            end_time = np.datetime64(datetime.datetime(end_year, end_month, 1, 0, 0, 0))
        else:
            end_time = obj.time[-1]
        return obj.sel(time=(obj.time >= start_time) * (obj.time <= end_time))

    
    raise TypeError('This function expects "cube_or_df" to be an iris Cube or a pandas DataFrame.')
    

def preprocess_model(ds):    
    filename = ds['burntarea-total'].encoding['source']
    modelname = filename.split('/')[-1].split('_gswp3')[0]
    ds = preprocess_time(ds)
    ds = ds.rename({'burntarea-total': modelname})
    warnings.simplefilter(action="default", category=UserWarning)
    return ds[modelname]   
    

def preprocess(ds, variables, start_year=1901, end_year=2020, start_month=1):
    if 'spei' in ds.keys():
        ds_name = ds.attrs['Id'][-9:-3] # Probably wrong
        ds = ds.rename({'spei': ds_name})
    if 'percentage_of_area_burned' in ds.keys():
        ds = xr.where(ds>101, np.nan, ds) # All ocean values are 9.9e36, replace by nan
        ds = xr.where(ds>100, 100, ds) # There's a few values that go over 100 (max is 100.3)
    if 'fwi' in ds.keys():
        ds['fwi'] = ds.fwi.fillna(0)
    if 'density' in ds.keys():
        ds = ds.rename({'density': 'lightning-density'})
        
    dropvars = [var for var in list(ds.keys()) if (var.endswith('_bnds') or var=='latitude_longitude' or var not in variables)]
    
    croptypes = [key for key in ds.keys() if key.startswith('c3') or key.startswith('c4')]
    if croptypes:
        ds['cropland'] = sum(ds[croptype] for croptype in croptypes)
        dropvars + croptypes
    
    ds = ds.drop_vars(dropvars)
    ds = preprocess_time(ds)
    ds = constrain_time(ds, start_year=start_year, end_year=end_year, start_month=start_month)
    
    if ds.lat[0] < ds.lat[-1]:
        ds = ds.reindex(lat=list(reversed(ds.lat)))
    
    return ds


def divide_by_100(x):
    return x/100

def log_transform(arr, epsilon=1e-6):
    return np.log1p(arr + epsilon) - epsilon

def log_transform_scaled(arr, epsilon_magnitude=6):
    return (np.log10(arr + 1/10**epsilon_magnitude) + epsilon_magnitude)/epsilon_magnitude

def log_inverse(arr, epsilon=1e-6):
    return np.expm1(arr) - epsilon

def log_inverse_scaled(arr, epsilon_magnitude=6):
    return 10**(arr * epsilon_magnitude-epsilon_magnitude) - 1/10**epsilon_magnitude


def normalize_features(x):
    feature_mean = x.mean(dim=['sample', 'time'])
    feature_std = x.std(dim=['sample', 'time'])
    return (x - feature_mean)/feature_std


def RMSE(a, b):
    if type(a) == iris.cube.Cube:
        return math.sqrt(((a-b)**2).data.mean())
    return math.sqrt(((a-b)**2).mean())

def mask_region(cube, region_mask):
    return iris.util.mask_cube(cube, region_mask, in_place=False, dim=0)

def add_year_coord(cube):
    if not any([coord.long_name == 'year' for coord in cube.coords()]):
        iris.coord_categorisation.add_year(cube, 'time', name='year')
    return cube

def to_annual(obj, sum_or_mean='sum'):
    if type(obj) == iris.cube.Cube:
        obj = add_year_coord(obj) 
        if sum_or_mean == 'sum':
            return obj.aggregated_by(['year'], iris.analysis.SUM)
        elif sum_or_mean == 'mean':
            return obj.aggregated_by(['year'], iris.analysis.MEAN)
        else:
            raise ValueError('sum_or_mean should be "sum" or "mean".') 
    elif type(obj) in (pd.core.frame.DataFrame, pd.core.series.Series):
        # 1. Convert the index to year to_period('Y'), which turns it into a pandas.PeriodIndex.
        # 2. Get the year as int (.year, seaborn doesn't know how to handle a pandas.PeriodIndex)
        # 3. Group by year .groupby()
        grouped_yearly = obj.groupby(obj.index.to_period('Y').year)
        # 4. Take the sum or mean of each group. Do not use NaNs (a sum or mean of NaNs equals 0 (undesirable))
        if sum_or_mean == 'sum':
            return grouped_yearly.sum(numeric_only=True, min_count=1)
        elif sum_or_mean == 'mean':
            return grouped_yearly.mean(numeric_only=True)
        else:
            raise ValueError('sum_or_mean should be "sum" or "mean".')  
    raise TypeError('This function expects "cube_or_df" to be an iris Cube or a pandas DataFrame.')   


def to_global(cube, func='sum', area_weighted=False):
    if func.lower() == 'sum':
        func = iris.analysis.SUM
    elif func.lower() == 'mean':
        func = iris.analysis.MEAN
    # Transforms an iris cube into a timeseries (take the area-weighted sum over the longitude and latitude dimensions).
    coords = ('longitude', 'latitude')
    for coord in coords:
        if not cube.coord(coord).has_bounds():
            cube.coord(coord).guess_bounds()
    if area_weighted:
        weights = iris.analysis.cartography.area_weights(cube) # pixel size in m^2
        return cube.collapsed(coords, func, weights = weights)
    return cube.collapsed(coords, func)


def to_mha_cube(cube):
    # Transforms percentage to Mha
    new_cube = cube.copy()
    for coord in ('longitude', 'latitude'):
        if not new_cube.coords(coord)[0].has_bounds():
            new_cube.coords(coord)[0].guess_bounds()
        new_cube.coords(coord)[0].coord_system = cs
        new_cube.coords(coord)[0].units = 'degrees'    
    weights = iris.analysis.cartography.area_weights(new_cube) # pixel size in m^2
    new_cube.units = '1e6 hectare' # Mha
    return new_cube * (weights/10**12)

def to_mha(obj):
    if type(obj) == iris.cube.Cube:
        return to_mha_cube(obj)
    elif type(obj) == iris.cube.CubeList:
        result_cubelist = obj.copy()
        for idx, cube in enumerate(result_cubelist):
            cube_name = cube.long_name
            result_cubelist[idx] = to_mha_cube(cube)
            result_cubelist[idx].long_name = cube_name
        return result_cubelist
    else:
        raise TypeError('This function expects an iris Cube or CubeList.')

def to_perc_cube(cube):
    # Transforms Mha to percentage
    new_cube = cube.copy()
    weights = iris.analysis.cartography.area_weights(new_cube) # pixel size in m^2
    new_cube.units = '%'
    return new_cube / (weights/10**12)

def to_perc(obj):
    if type(obj) == iris.cube.Cube:
        return to_perc_cube(obj)
    elif type(obj) == iris.cube.CubeList:
        result_cubelist = obj.copy()
        for idx, cube in enumerate(result_cubelist):
            cube_name = cube.long_name
            result_cubelist[idx] = to_perc_cube(cube)
            result_cubelist[idx].long_name = cube_name
        return result_cubelist
    else:
        raise TypeError('This function expects an iris Cube or CubeList.')

def pixel_mean(cube):
    return cube.collapsed(('time'), iris.analysis.MEAN)        

def ds_to_cubelist(ds):
    cube_list = [] 
    time_unit = cf_units.Unit('days since 1901-01-01T00:00:00', calendar='standard')
    for modelname in list(ds.keys()):
        model_da = ds[modelname]
        model_da.attrs['standard_name'] = None
        cube = model_da.to_iris()
        cube.long_name = cube.var_name
        cube.var_name = 'burntarea-total'
        for coord in ('longitude', 'latitude'):
            cube.coords(coord)[0].guess_bounds()
            cube.coords(coord)[0].coord_system = cs
            cube.coords(coord)[0].units = 'degrees'        
        
        time_coord = cube.coords('time')[0]
        time_coord.long_name = None
        time_coord.units = time_coord.units.change_calendar('standard')
        date_values = time_coord.units.num2date(time_coord.points)
        new_time_values = time_unit.date2num(date_values)
        new_time_coord = iris.coords.DimCoord(new_time_values,
                                              standard_name='time',
                                              var_name='time',
                                              units=time_coord.units)
        cube.replace_coord(new_time_coord)
        cube.coords('time')[0].units = time_unit
        
        cube_list.append(cube)
    return iris.cube.CubeList(cube_list)

def to_dataset(cubelist):
    ds = xr.merge([xr.DataArray.from_iris(cube) for cube in cubelist])
    return ds

def folds_to_region(folds):
    return [regionname for fold in folds for regionname in fold]

def weighted_mean(da):
    lat1, lat2 = np.deg2rad(da.lat-0.25), np.deg2rad(da.lat+0.25)
    lon1, lon2 = np.deg2rad(da.lon-0.25), np.deg2rad(da.lon+0.25)
    weights = 6371229**2*((lon1-lon2)*(np.sin(lat1) - np.sin(lat2)))
    return da.weighted(weights).mean(dim=['stacked_lat_lon'])#/weights.sum()

def f1_recall_precision(outputs, targets):
    true_pos = torch.count_nonzero((outputs>0)[targets>0])
    true_neg = torch.count_nonzero((outputs<=0)[targets<=0])
    false_pos = torch.count_nonzero((outputs>0)[targets<=0])
    false_neg = torch.count_nonzero((outputs<=0)[targets>0])

    if not false_pos: # edge case if false_pos is 0, prevent division by 0
        precision = 1
    else:
        precision = true_pos/(true_pos+false_pos)
        precision = precision
    
    if not false_neg: # edge case, prevent division by 0
        recall = 1
    else:
        recall = true_pos/(true_pos+false_neg)
        recall = recall
    
    if not false_pos and not false_neg: # edge case, prevent division by 0
        f1 = 1
    else:
        f1 = (2*true_pos)/(2*true_pos + false_pos + false_neg)
        f1 = f1
    return f1, recall, precision
    

def nse(outputs, targets, epsilon=1e-1):
    # Add small error to variance, 0.1?
    mse = nn.functional.mse_loss(outputs.squeeze(dim=-1), targets.squeeze(dim=-1), reduction='none').mean(dim=1)
    std_obs = torch.std(targets.squeeze(dim=-1), dim=1)
    std_obs += epsilon
    nse = (mse/std_obs**2).mean()
    return nse

def kge(outputs, targets, epsilon=1e-2):
    mean_target = targets.squeeze(dim=-1).mean(dim=1)
    mean_pred = outputs.squeeze(dim=-1).mean(dim=1)
    std_target = torch.std(targets.squeeze(dim=-1), dim=1)
    std_pred = torch.std(outputs.squeeze(dim=-1), dim=1)
    
    alpha = torch.where(std_target==0, 1, std_pred/std_target).mean() # Variance ratio, ideally 1
    beta = torch.where(std_target==0, (mean_pred-mean_target)/epsilon, (mean_pred-mean_target)/std_target).mean() # Bias, ideally 0
    r = pearson_corrcoef(outputs.squeeze(dim=-1), targets.squeeze(dim=-1)).mean() # Correlation, ideally 1
    kge = torch.sqrt((r-1)**2 + (alpha-1)**2) + (beta)**2 # Returning a modified version; best to worst is 0 to +inf (instead of 1 to -inf).
    return kge, alpha, beta, r


def Binary_and_MSELoss(outputs, targets, bce_loss_fn, alpha=0.5, lambda_factor=1):
    binary_output = outputs[:, :, 0].unsqueeze(-1)
    regression_output = outputs[:, :, 1].unsqueeze(-1)
        
    # Create binary target
    binary_target = targets > 0
    
    # Calculate binary loss
    classification_loss = bce_loss_fn(binary_output, binary_target.float())  

    # Calculate regression loss for positive samples  
    mask = targets > 0  
    if mask.any():  # Avoid division by zero
        regression_loss = nn.MSELoss(reduction='sum')(regression_output[mask], targets[mask])
    else:
        regression_loss = torch.tensor(0.0)  # No non-zero targets in this batch
    #regression_output = (binary_classification_target>0).float() * regression_output
    #regression_loss = nn.MSELoss(reduction='mean')(regression_output, targets)   
    
    # Combine the losses
    total_loss = alpha*classification_loss + (1-alpha) * lambda_factor * regression_loss
    
    log_params = {
        'loss': total_loss.detach(),
        'classification_loss': classification_loss.detach(),
        'regression_loss': (lambda_factor * regression_loss).detach()}
    
    log_params['NSE'] = nse(regression_output, targets)
    #log_params['KGE'], log_params['KGE_alpha'], log_params['KGE_beta'], log_params['KGE_r'] = kge(regression_output, targets)
    log_params['F1-score'], log_params['recall'], log_params['precision'] = f1_recall_precision(binary_output, targets)

    log_params = {key: value.detach() if isinstance(value, torch.Tensor) else value for key, value in log_params.items()}
    
    return total_loss, log_params

def MSELoss(outputs, targets, lambda_factor=1):
    total_loss = lambda_factor * nn.functional.mse_loss(outputs, targets)

    log_params = {
        'loss': total_loss.detach(),
        }
    log_params['NSE'] = nse(outputs, targets)
    #log_params['KGE'], log_params['KGE_alpha'], log_params['KGE_beta'], log_params['KGE_r'] = kge(outputs, targets)
    log_params['F1-score'], log_params['recall'], log_params['precision'] = f1_recall_precision(outputs, targets)

    log_params = {key: value.detach() if isinstance(value, torch.Tensor) else value for key, value in log_params.items()}

    return total_loss, log_params 

def DoubleMSELoss(outputs, targets, lambda_factor=1):
    total_loss = nn.functional.mse_loss(outputs, targets) + lambda_factor * nn.functional.mse_loss(outputs.sum(axis=1), targets.sum(axis=1))

    log_params = {
        'loss': total_loss.detach(),
        }
    log_params['NSE'] = nse(outputs, targets)
    #log_params['KGE'], log_params['KGE_alpha'], log_params['KGE_beta'], log_params['KGE_r'] = kge(outputs, targets)
    log_params['F1-score'], log_params['recall'], log_params['precision'] = f1_recall_precision(outputs, targets)

    log_params = {key: value.detach() if isinstance(value, torch.Tensor) else value for key, value in log_params.items()}

    return total_loss, log_params 

def MSE_and_VarLoss(outputs, targets, lambda_factor=1):
    mse_loss = nn.functional.mse_loss(outputs, targets)

    var_pred = torch.var(outputs, dim=1, unbiased=False)
    var_true = torch.var(targets, dim=1, unbiased=False)
    
    # Compute the variance matching loss (MSE between variances)
    var_loss = nn.functional.mse_loss(var_pred, var_true)
    total_loss = mse_loss + lambda_factor*var_loss

    log_params = {
        'loss': total_loss.detach(),
        'Variance_loss': (var_loss*lambda_factor).detach(),
        'MSE_loss': mse_loss.detach()
        }
    log_params['NSE'] = nse(outputs, targets)
    #log_params['KGE'], log_params['KGE_alpha'], log_params['KGE_beta'], log_params['KGE_r'] = kge(outputs, targets)
    log_params['F1-score'], log_params['recall'], log_params['precision'] = f1_recall_precision(outputs, targets)

    log_params = {key: value.detach() if isinstance(value, torch.Tensor) else value for key, value in log_params.items()}

    return total_loss, log_params 

def CustomLoss(outputs, targets, metric_name, lambda_factor=1):
    outputs = outputs[:, :, 1].unsqueeze(-1)
    log_params = {}
    log_params['NSE'] = nse(outputs, targets)
    #log_params['KGE'], log_params['KGE_alpha'], log_params['KGE_beta'], log_params['KGE_r'] = kge(outputs, targets)
    log_params['F1-score'], log_params['recall'], log_params['precision'] = f1_recall_precision(outputs, targets)

    log_params['loss'] = log_params[metric_name]
    loss = log_params[metric_name].clone()
    log_params = {key: value.detach() if isinstance(value, torch.Tensor) else value for key, value in log_params.items()}
    return loss, log_params

INVERSE_FUNCS = {divide_by_100: lambda x: x.multiply(100),
                    log_transform: log_inverse,
                    log_transform_scaled: log_inverse_scaled
                         }



