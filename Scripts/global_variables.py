import os
from pathlib import Path
import numpy as np
import xarray as xr
import iris
from iris import coord_systems, util, cube
import regionmask
from pathlib import Path

data_path = Path('/vscmnt/brussel_pixiu_data/_data_brussel/vo/000/bvo00012/vsc10262/Wildfires/WP1_Long_Term_BA_Reconstruction/Data/')
figures_path = os.path.join(data_path.parent, 'Figures')
model_path = os.path.join(data_path.parent, 'Models')
supplementary_data_path = os.path.join(data_path.parent, 'Supplementary_Data')
supplementary_output_path = os.path.join(data_path.parent, 'Supplementary_Output')
SEED = int(int.from_bytes("They're taking the hobbits to Isengard!".encode('utf-8'), byteorder='big')%(2**32-1))
isimip_fire_path = Path('/vscmnt/brussel_pixiu_data/_data_brussel/vo/000/bvo00012/data/dataset/ISIMIP/ISIMIP3a/OutputData/fire/')

cs = iris.coord_systems.GeogCS(6371229)
lon = np.arange(-179.75, 180, 0.5)
lat = np.arange(-89.75, 90, 0.5)
AR6_names = ['GIC', 'NWN', 'NEN', 'WNA', 'CNA', 'ENA', 'NCA', 'SCA', 'CAR', 'NWS', 'NSA', 'NES', 'SAM', 'SWS', 'SES', 'SSA', 'NEU', 'WCE', 'EEU', 'MED', 'SAH', 'WAF', 'CAF', 'NEAF', 'SEAF', 'WSAF', 'ESAF', 'MDG', 'RAR', 'WSB', 'ESB', 'RFE', 'WCA', 'ECA', 'TIB', 'EAS', 'ARP', 'SAS', 'SEA', 'NAU', 'CAU', 'EAU', 'SAU', 'NZ']
AR6_mask = ~regionmask.defined_regions.ar6.land.mask_3D(lon, lat) # Create the mask on the 0.5 by 0.5 degree scale and reverse it (tilde) so the region of interest = False
# Conversion to np.array necessary as cube.data.data is expected to be a numpy array, not an xarray.DataArray
# All the metadata for the dim_coords is also required to match with the obs/model dim_coords

AR6_idx_name = {idx: name for idx, name in enumerate(AR6_names)}
AR6_name_idx = {name: idx for idx, name in enumerate(AR6_names)}

AR6_masks = {idx: 
             iris.util.reverse(
                 iris.cube.Cube(np.array(AR6_mask[idx]), 
                            dim_coords_and_dims=[(iris.coords.DimCoord(AR6_mask[idx].coords['lat'], standard_name = 'latitude', units='degrees', long_name='Latitude', var_name='lat', coord_system=cs), 0), 
                                                 (iris.coords.DimCoord(AR6_mask[idx].coords['lon'], standard_name = 'longitude', units='degrees', long_name='Longitude', var_name='lon', coord_system=cs), 1)], 
                            var_name=name),
                 'latitude')
             for idx, name in enumerate(AR6_names)}

del AR6_masks[AR6_name_idx['GIC']]

for mask in AR6_masks.values():
    mask.coords('latitude')[0].guess_bounds()
    mask.coords('longitude')[0].guess_bounds()



gfed_names = ['OCEAN', 'BONA', 'TENA', 'CEAM', 'NHSA', 'SHSA', 'EURO', 'MIDE', 'NHAF', 'SHAF', 'BOAS', 'CEAS', 'SEAS', 'EQAS', 'AUST']

gfed_idx_name = {idx: name for idx, name in enumerate(gfed_names)}
gfed_name_idx = {name: idx for idx, name in enumerate(gfed_names)}

gfed_mask = xr.open_dataset(r'/vscmnt/brussel_pixiu_data/_data_brussel/vo/000/bvo00012/vsc10262/Wildfires/WP1_Long_Term_BA_Reconstruction/Supplementary_Data/base_regions_upscaled.nc')

for idx in gfed_idx_name:
    region = gfed_idx_name[idx]
    gfed_mask[region] = gfed_mask.__xarray_dataarray_variable__ != idx

gfed_mask = gfed_mask.drop_vars('__xarray_dataarray_variable__')

gfed_masks = {idx: 
                 iris.cube.Cube(np.array(gfed_mask[name]), 
                            dim_coords_and_dims=[(iris.coords.DimCoord(gfed_mask.coords['lat'], standard_name = 'latitude', units='degrees', long_name='Latitude', var_name='lat', coord_system=cs), 0), 
                                                 (iris.coords.DimCoord(gfed_mask.coords['lon'], standard_name = 'longitude', units='degrees', long_name='Longitude', var_name='lon', coord_system=cs), 1)], 
                            var_name=name)
              for idx, name in enumerate(gfed_names)
             }

del gfed_masks[gfed_name_idx['OCEAN']]

for mask in gfed_masks.values():
    mask.coords('latitude')[0].guess_bounds()
    mask.coords('longitude')[0].guess_bounds()