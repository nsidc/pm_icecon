import datetime as dt
from pathlib import Path

import cartopy.crs as ccrs
from matplotlib import pyplot as plt
import numpy as np
import xarray as xr

from cdr_amsr2.fetch import au_si25


EXAMPLE_BT_DIR = Path('/share/apps/amsr2-cdr/cdr_testdata/bt_amsru_regression/')
EXAMPLE_BT_NC = EXAMPLE_BT_DIR / 'NH_20200101_py_NRT_amsr2.nc'

# TODO: these stolen from `seaice`'s `image` submodules. Eventually we'll want
# to be able to generate 'standard' images using the `seaice` library, but that
# will take some additional work.
COLORS = [
    '#093c70',   # 0-5
    '#093c70',   # 5-10
    '#093c70',   # 10-15
    '#137AE3',   # 15-20
    '#1684EB',   # 20-25
    '#178CF2',   # 25-30
    '#1994F9',   # 30-35
    '#1A9BFC',   # 35-40
    '#23A3FC',   # 40-45
    '#31ABFC',   # 45-50
    '#45B4FC',   # 50-55
    '#57BCFC',   # 55-60
    '#6AC4FC',   # 60-65
    '#7DCCFD',   # 65-70
    '#94D5FD',   # 70-75
    '#A8DCFD',   # 75-80
    '#BCE4FE',   # 80-85
    '#D0ECFE',   # 85-90
    '#E4F4FE',   # 90-95
    '#F7FCFF',   # 95-100
    '#e9cb00',  # 110missing
    '#777777',   # 120land
]

COLORBOUNDS = [
    0.,
    5.,
    10.,
    15.,
    20.,
    25.,
    30.,
    35.,
    40.,
    45.,
    50.,
    55.,
    60.,
    65.,
    70.,
    75.,
    80.,
    85.,
    90.,
    95.,
    100.0001,
    110.001,
    120.001,
]

N_PROJ = ccrs.Stereographic(
    central_latitude=90.0,
    central_longitude=-45.0,
    false_easting=0.0,
    false_northing=0.0,
    true_scale_latitude=70,
    globe=None,
)


def get_example_output() -> xr.Dataset:
    """Get the example AMSR2 output from our python code.

    * Flip the data so that North is 'up'.
    * Scale the data by 10 and round to np.uint8 dtype.
    """
    example_ds = xr.open_dataset(EXAMPLE_BT_NC)
    # flip the image to be 'right-side' up
    example_ds = example_ds.reindex(y=example_ds.y[::-1], x=example_ds.x)

    # scale the data by 10 and convert to int
    example_ds['conc'] = example_ds.conc / 10

    # Round the data as integers.
    example_ds['conc'] = (example_ds.conc + 0.5).astype(np.uint8)

    return example_ds


def save_n_conc_image(conc_array: xr.DataArray, filepath: Path) -> None:
    """Create an image representing the N. hemisphere conc field."""
    conc_array.plot.imshow(
        ax=plt.axes((0, 0, 1, 1), projection=N_PROJ),
        colors=COLORS,
        levels=COLORBOUNDS,
        add_colorbar=False,
        add_labels=False
    )
    plt.savefig(filepath, bbox_inches='tight', pad_inches=0.05)

    plt.clf()
    plt.cla()
    plt.close()


def get_au_si25_bt_conc() -> xr.DataArray:
    date = dt.date(2020, 1, 1)
    ds = au_si25._get_au_si25_data_fields(
        base_dir=Path('/ecs/DP1/AMSA/AU_SI25.001/'),
        date=date,
        hemisphere='north',
    )

    # flip the image to be 'right-side' up
    ds = ds.reindex(YDim=ds.YDim[::-1], XDim=ds.XDim)
    ds = ds.rename({'YDim': 'y', 'XDim': 'x'})

    nt_conc = ds.SI_25km_NH_ICECON_DAY
    diff = ds.SI_25km_NH_ICEDIFF_DAY
    bt_conc = nt_conc + diff

    return bt_conc



if __name__ == '__main__':
    # Get and save an image of the NH example data produced by our python code.
    example_ds = get_example_output()
    save_n_conc_image(example_ds.conc, Path('/tmp/NH_20200101_py_amsr2.png'))

    # Do the same for the bootstrap concentration field that comes with the
    # AU_SI25 data.
    au_si25_conc = get_au_si25_bt_conc()
    save_n_conc_image(au_si25_conc, Path('/tmp/NH_20200101_au_si25_amsr2.png'))

    # Do a difference between the two images.
    diff = example_ds.conc - au_si25_conc
    plt.clf()
    plt.cla()
    plt.close()
    diff.plot.imshow(
        ax=plt.axes((0, 0, 1, 1), projection=N_PROJ),
        add_colorbar=True,
    )
    plt.savefig(
        '/tmp/NH_20200101_diff_amsr2.png',
        bbox_inches='tight',
        pad_inches=0.05,
    )
