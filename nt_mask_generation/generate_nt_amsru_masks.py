"""Extract the NH and SH land masks for 25km and 12.5km.

Note: The input files for this routine are:
    NT ancillary files:
        shoremap_north_25
        shoremap_south_25
        SSMI8_monavg_min_con
        SSMI_monavg_min_con_s
    Sample AMSR_U2_L3 seaice files:
        AMSR_U2_L3_SeaIce12km_B04_20210101
        AMSR_U2_L3_SeaIce25km_B04_20210101

The procedure is to:
    - Determine the land masks by reading in NH and SH, 12.5/25km
      concentration fields from sample AMSR_U2_L3 seaice files
    - Use grid layout to determine -- from the land masks:
      - inner land
      - coast:  land orthogonally adjacent to water
      - shore:  water diagonally adjacent to coast
      - near shore: water orthogonally adjacent to shore
      - far shore: water orthogonally adjacent to near shore
    - Use the 25km minic -- = "minimum ice concentration" -- files
      from the original NASA Team code base to determine a minimum
      ice concentration field
      - consider pixels originally masked as "land" in the NT files
        to have a minic of 1000 (meaning 100% ice)

    - Write out the shoremap and minic fields for each NH/SH, 12/25 combo
"""

import numpy as np
from h5py import File
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import shift
from scipy.signal import convolve2d

# Read conc fields
ifn = './AMSR_U2_L3_SeaIce25km_B04_20210101.he5'
fid = File(ifn, 'r')
conc = {}
conc['nh25'] = np.array(fid['HDFEOS']['GRIDS']['NpPolarGrid25km']['Data Fields']['SI_25km_NH_ICECON_DAY'])  # noqa
conc['sh25'] = np.array(fid['HDFEOS']['GRIDS']['SpPolarGrid25km']['Data Fields']['SI_25km_SH_ICECON_DAY'])  # noqa
fid = None

ifn = './AMSR_U2_L3_SeaIce12km_B04_20210101.he5'
fid = File(ifn, 'r')
conc['nh12'] = np.array(fid['HDFEOS']['GRIDS']['NpPolarGrid12km']['Data Fields']['SI_12km_NH_ICECON_DAY'])  # noqa
conc['sh12'] = np.array(fid['HDFEOS']['GRIDS']['SpPolarGrid12km']['Data Fields']['SI_12km_SH_ICECON_DAY'])  # noqa
fid = None

print('Conc field shapes:')
for key in conc.keys():
    print(f'  conc_{key}: {conc[key].dtype}  {conc[key].shape}')

# Read in original NT fields
goddard_land = {}

# NH
goddard_land['raw_nh25'] = np.fromfile('./shoremap_north_25', dtype='>i2')[150:].reshape(448, 304)  # noqa
goddard_land['nh25'] = goddard_land['raw_nh25'].astype(np.uint8)
print(f'values in goddard_land: {np.unique(goddard_land["nh25"])}')

goddard_minic = {}
goddard_minic['raw_nh25'] = np.fromfile('./SSMI8_monavg_min_con', dtype='>i2')[150:].reshape(448, 304)  # noqa
print(f'goddard_minic: min: {goddard_minic["raw_nh25"].min()}')
print(f'goddard_minic: max: {goddard_minic["raw_nh25"].max()}')
goddard_minic['nh25'] = np.zeros((448, 304), dtype=np.int16)
goddard_minic['nh25'][:] = goddard_minic['raw_nh25']

# SH
goddard_land['raw_sh25'] = np.fromfile('./shoremap_south_25', dtype='>i2')[150:].reshape(332, 316)  # noqa
goddard_land['sh25'] = goddard_land['raw_sh25'].astype(np.uint8)
print(f'values in goddard_land: {np.unique(goddard_land["sh25"])}')

goddard_minic['raw_sh25'] = np.fromfile('./SSMI_monavg_min_con_s', dtype='>i2')[150:].reshape(332, 316)  # noqa
print(f'goddard_minic: min: {goddard_minic["raw_sh25"].min()}')
print(f'goddard_minic: max: {goddard_minic["raw_sh25"].max()}')
goddard_minic['sh25'] = np.zeros((332, 316), dtype=np.int16)
goddard_minic['sh25'][:] = goddard_minic['raw_sh25']

# Test same between Goddard NT and AMSRU
is_land = {}

# NH
is_land['godd_nh25'] = (goddard_land['nh25'] == 1) | (goddard_land['nh25'] == 2)  # noqa
is_land['amsru_nh25'] = (conc['nh25'] == 120)
is_land['amsru_nh12'] = (conc['nh12'] == 120)

print(f'land_godd_nh25 same as land_amsru_nh25: {np.all(is_land["godd_nh25"] == is_land["amsru_nh25"])}')  # noqa

# SH
is_land['godd_sh25'] = (goddard_land['sh25'] == 1) | (goddard_land['sh25'] == 2)  # noqa
is_land['amsru_sh25'] = (conc['sh25'] == 120)
is_land['amsru_sh12'] = (conc['sh12'] == 120)

# Use interpolation to generate NH minic field
# For the purposes of minic-field interpolation, cause land to be 1000
# Then, overwrite land with the same value of '1' as in the orig field
# NH
filled_minic_nh25 = goddard_minic['nh25'].copy()
filled_minic_nh25[is_land['godd_nh25']] = 1000
interp_func = RectBivariateSpline(
    np.arange(0, 448),
    np.arange(0, 304),
    goddard_minic['nh25'],
    kx=1,
    ky=1)
goddard_minic['nh12'] = interp_func(np.arange(0, 448, 0.5), np.arange(0, 304, 0.5)).astype(np.int16)  # noqa
goddard_minic['nh12'][is_land['amsru_nh12']] = 1

# SH
# "Fix" the 25km SH mask by filling in "land" with 1000
filled_minic_sh25 = goddard_minic['sh25'].copy()
filled_minic_sh25[is_land['godd_sh25']] = 1000
filled_minic_sh25[is_land['amsru_sh25']] = 1

# continue calculating amsru_12...
interp_func = RectBivariateSpline(
    np.arange(0, 332),
    np.arange(0, 316),
    goddard_minic['sh25'],
    kx=1,
    ky=1)
goddard_minic['sh12'] = interp_func(np.arange(0, 332, 0.5), np.arange(0, 316, 0.5)).astype(np.int16)  # noqa
goddard_minic['sh12'][is_land['amsru_sh12']] = 1

for key in goddard_minic.keys():
    if 'raw' not in key:
        # ydim, xdim = goddard_minic[key].shape
        # ofn = f'minic_amsru_{key}_{xdim}x{ydim}.dat'
        ofn = f'minic_amsru_{key}.dat'
        goddard_minic[key].tofile(ofn)
        print(f'Wrote: {ofn}  {goddard_minic[key].dtype}  {goddard_minic[key].shape}')  # noqa

# Now, compute the shoremap files from the land file
# Note: Doing this revealed that there are a few errors
#       in the shoremap_north_25 field.
for key in [key for key in is_land.keys() if 'amsru' in key]:
    # Find land and give it a value of 1
    field = np.zeros(is_land[key].shape, dtype=np.uint8)
    field[is_land[key]] = 1

    # Find coast, and give it a value of 2
    left = shift(field, (0, -1), order=0, mode='nearest')
    right = shift(field, (0, 1), order=0, mode='nearest')
    up = shift(field, (-1, 0), order=0, mode='nearest')
    down = shift(field, (1, 0), order=0, mode='nearest')

    is_coast = (field == 1) & \
        ((left == 0) | (right == 0) | (up == 0) | (down == 0))
    field[is_coast] = 2

    # Now, expand
    # Convolve the field with a 3x3 'ones' matrix to get all
    # diagonally connected points
    convolved = convolve2d(
        field,
        [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
        mode='same',
        boundary='symm')
    is_shore = (convolved > 0) & (field == 0)
    field[is_shore] = 3

    # Expand to near_shore, *not* diagonally connected
    convolved = convolve2d(
        field,
        [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
        mode='same',
        boundary='symm')
    is_near = (convolved > 0) & (field == 0)
    field[is_near] = 4

    # Expand to far_shore, *not* diagonally connected
    convolved = convolve2d(
        field,
        [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
        mode='same',
        boundary='symm')
    is_far = (convolved > 0) & (field == 0)
    field[is_far] = 5

    # ydim, xdim = field.shape
    # ofn = f'shoremap_{key}_{xdim}x{ydim}.dat'
    ofn = f'shoremap_{key}.dat'
    field.tofile(ofn)
    print(f'Wrote: {ofn}  {field.dtype}  {field.shape}')

    """
    # This section compares the here-derived shoremap with that
    #   provided in the original NT code package.  There are a
    #   few small errors in the original file.
    if key == 'amsru_nh25':
        print('Comparing to shoremap_north_25:')
        is_same_shoremap_nh25 = \
            np.all(field == goddard_land["nh25"])
        print(f'  field == shoremap_north_25: {is_same_shoremap_nh25}')
        if not is_same_shoremap_nh25:
            where_not_same = np.where(field != goddard_land['nh25'])
            for i, j in zip(where_not_same[1], where_not_same[0]):
                print(f'  Not same at: ({i}, {j})')
    """
