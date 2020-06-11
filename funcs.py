from netCDF4 import Dataset
from pyproj import Proj, transform
import numpy as np
import warnings

def get(string,grid_dir):

    """"Returns a 361x361 grid of lon lat values for the 25 km EASE grid.
    These are taken from a netcdf file which is included in this repo."""

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

    """Converts EASE grid coordinates to WGS 84. Can take a single point or two lists"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        EASE_Proj = Proj(init='epsg:3408')
        WGS_Proj = Proj(init='epsg:4326')
        lon,lat = transform(EASE_Proj,WGS_Proj,x,y)
        return(lon,lat)

def lonlat_to_xy(lon,lat):

    """Converts WGS 84 coords to EASE grid. Can take a single point or two lists"""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        EASE_Proj = Proj(init='epsg:3408')
        WGS_Proj = Proj(init='epsg:4326')
        x,y = transform(WGS_Proj,EASE_Proj,lon,lat)
        return(x,y)


def get_day_vectors(date_obj):

    """Returns a 2-part dictionary of u and v vectors for a given date, on 361x361 25 km EASE grid"""

    day_of_year = date_obj.timetuple().tm_yday

    data = Dataset('/home/robbie/Dropbox/Data/IMV/icemotion_daily_nh_25km_20160101_20161231_v4.1.nc')

    data_for_day = {'u': data['u'][day_of_year - 1],
                    'v': data['v'][day_of_year - 1]}

    return (data_for_day)


def one_iteration(point, field, tree, timestep):

    """Iterates a point based on its position in an ice motion field. Must be passed a pre-calculated KDTree
    of the field (which saves time). Timestep must be in seconds, field vectors must be in cm/s. If the point's
    nearest velocity value is nan (representing open water), then it returns a nan point (np.nan, np.nan)"""

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