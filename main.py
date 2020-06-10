from netCDF4 import Dataset
import datetime
from funcs import lonlat_to_xy, get, one_iteration
from scipy.spatial import KDTree
import numpy as np
import sys
from tqdm import trange

# What year?

year = int(sys.argv[1])

data_dir = '/home/robbie/Dropbox/Data/IMV'

# What day do you want to start?

start_day = datetime.date(year=year,month=8,day=31)
start_day_of_year = start_day.timetuple().tm_yday

# Get dataset

data_16 = Dataset(f'{data_dir}/icemotion_daily_nh_25km_{year}0101_{year}1231_v4.1.nc')
data_17 = Dataset(f'{data_dir}/icemotion_daily_nh_25km_{year+1}0101_{year+1}1231_v4.1.nc')

# Get EASE_lons & EASE_lats

EASE_lons = get('lon')
EASE_lats = get('lat')

EASE_xs, EASE_ys = lonlat_to_xy(EASE_lons.ravel(), EASE_lats.ravel())

EASE_coords = np.array

EASE_tree = KDTree(list(zip(EASE_xs, EASE_ys)))

EASE_xs_grid = EASE_xs.reshape((361,361))
EASE_ys_grid = EASE_ys.reshape((361,361))

start_x, start_y = EASE_xs.ravel(), EASE_ys.ravel()

all_u16, all_v16 = np.array(data_16['u']), np.array(data_16['v'])
all_u17, all_v17 = np.array(data_17['u']), np.array(data_17['v'])

all_u = np.ma.concatenate((all_u16, all_u17), axis=0)
all_v = np.ma.concatenate((all_v16, all_v17), axis=0)

all_u = np.ma.masked_where(all_u==-9999.0, all_u)
all_u = np.ma.filled(all_u, np.nan)
all_v = np.ma.masked_where(all_v==-9999.0, all_v)
all_v = np.ma.filled(all_v, np.nan)

#######################################################

tracks_array = np.full((2,250,70_000), np.nan)

end_date = datetime.date(year=year+1,month=4,day=30)

data_for_start_day = {'u':all_u[start_day_of_year-1],
                      'v':all_v[start_day_of_year-1]}

u_field = data_for_start_day['u'].ravel()

# Select points on ease grid with valid velocity data on day 0

valid_start_x = start_x[~np.isnan(u_field)]
valid_start_y = start_y[~np.isnan(u_field)]

valid_points = list(zip(valid_start_x, valid_start_y))

exit()

for day_num in trange(0, 700):

    valid_points_x = list(zip(*valid_points))[0]
    valid_points_y = list(zip(*valid_points))[1]

    tracks_array[0, day_num, :len(valid_points_x)] = valid_points_x
    tracks_array[1, day_num, :len(valid_points_y)] = valid_points_y

    date = start_day + datetime.timedelta(days=day_num)

    day_of_year = date.timetuple().tm_yday

    print(f'Day_num: {day_num}, Total tracks: {len(valid_points)}')

    # Get the ice motion field for that day

    data_for_day = {'u': all_u[start_day_of_year + day_num - 1],
                    'v': all_v[start_day_of_year + day_num - 1]}

    # Update points

    updated_points = [one_iteration(point,
                                    data_for_day,
                                    EASE_tree,
                                    24 * 60 * 60) if point != (np.nan, np.nan) else point for point in valid_points]

    # Make list of points that are still alive

    clean_points = [point for point in updated_points if point != (np.nan, np.nan)]

    print(f'Tracks killed: {len(updated_points) - len(clean_points)}')

    # Create new parcels in gaps
    # Make a decision tree for the track field

    track_tree = KDTree(clean_points)

    # Identify all points of ease_grid with valid values

    u_field = np.ma.masked_values(data_for_day['u'].ravel(), np.nan)

    valid_x, valid_y = start_x[~np.isnan(u_field)], start_y[~np.isnan(u_field)]

    # Iterate through all valid points to identify gaps using the tree

    new_points = []

    for point in zip(valid_x, valid_y):
        distance, index = track_tree.query(point)

        if distance > 20_000:  # Initiate new track
            new_points.append(point)

    print(f'Tracks added: {len(new_points)}')

    # Add newly intitiated tracks to other tracks

    valid_points = updated_points + new_points

    if (len(valid_points) > 70_000) or (date > end_date): break

np.save(f'tracks_array_{year}.npy', tracks_array)