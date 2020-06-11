from netCDF4 import Dataset
from pyproj import Proj, transform
import numpy as np
import warnings

def get(string,grid_dir):

    path_grid = f'{grid_dir}grid.nc'

    if string == 'lon':
        grid_data = Dataset(path_grid)
        lon = np.array(grid_data.variables["lon"])
        return(lon)
    elif string == 'lat':
        grid_data = Dataset(path_grid)
        lat = np.array(grid_data.variables["lat"])
        return(lat)

def xy_to_lonlat(x,y):
    EASE_Proj = Proj(init='epsg:3408')
    WGS_Proj = Proj(init='epsg:4326')
    lon,lat = transform(EASE_Proj,WGS_Proj,x,y)
    return(lon,lat)

def lonlat_to_xy(lon,lat):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        EASE_Proj = Proj(init='epsg:3408')
        WGS_Proj = Proj(init='epsg:4326')
        x,y = transform(WGS_Proj,EASE_Proj,lon,lat)
        return(x,y)


def get_day_vectors(date_obj):
    year, month, day = date_obj.year, date_obj.month, date_obj.day

    day_of_year = date_obj.timetuple().tm_yday

    data = Dataset('/home/robbie/Dropbox/Data/IMV/icemotion_daily_nh_25km_20160101_20161231_v4.1.nc')

    data_for_day = {'u': data['u'][day_of_year - 1],
                    'v': data['v'][day_of_year - 1]}

    return (data_for_day)


def one_iteration(point, field, tree, timestep):
    distance, index = tree.query(point)

    u_vels, v_vels = np.array(field['u']) / 100, np.array(field['v']) / 100

    u_vels = np.ma.masked_where(u_vels == -99.99, u_vels)
    u_vels = np.ma.masked_values(u_vels, np.nan)

    u_vel, v_vel = u_vels.ravel()[index], v_vels.ravel()[index]

    if np.isnan(u_vel):
        #         print('Failed')
        return ((np.nan, np.nan))

    else:

        u_disp, v_disp = u_vel * timestep, v_vel * timestep

        new_position = (point[0] + u_disp, point[1] + v_disp)

        return (new_position)