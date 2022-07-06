"""One-off script to produce visualizations for the NOAA stakeholders meeting.

Meeting took place on June 30th, 2022 and went well!

This code is a bit of a mess, but may serve as reference for future
visualization code.
"""
import datetime as dt
from pathlib import Path

import cartopy.crs as ccrs
import numpy as np
import xarray as xr
from matplotlib import pyplot as plt

from cdr_amsr2.constants import PACKAGE_DIR
from cdr_amsr2.fetch import au_si25

EXAMPLE_BT_DIR = Path('/share/apps/amsr2-cdr/cdr_testdata/bt_amsru_regression/')
EXAMPLE_BT_NC = EXAMPLE_BT_DIR / 'NH_20200101_py_NRT_amsr2.nc'

# TODO: these stolen from `seaice`'s `image` submodules. Eventually we'll want
# to be able to generate 'standard' images using the `seaice` library, but that
# will take some additional work.
COLORS = [
    '#093c70',  # 0-5
    '#093c70',  # 5-10
    '#093c70',  # 10-15
    '#137AE3',  # 15-20
    '#1684EB',  # 20-25
    '#178CF2',  # 25-30
    '#1994F9',  # 30-35
    '#1A9BFC',  # 35-40
    '#23A3FC',  # 40-45
    '#31ABFC',  # 45-50
    '#45B4FC',  # 50-55
    '#57BCFC',  # 55-60
    '#6AC4FC',  # 60-65
    '#7DCCFD',  # 65-70
    '#94D5FD',  # 70-75
    '#A8DCFD',  # 75-80
    '#BCE4FE',  # 80-85
    '#D0ECFE',  # 85-90
    '#E4F4FE',  # 90-95
    '#F7FCFF',  # 95-100
    '#e9cb00',  # 110missing
    '#777777',  # 120land
]

COLORBOUNDS = [
    0.0,
    5.0,
    10.0,
    15.0,
    20.0,
    25.0,
    30.0,
    35.0,
    40.0,
    45.0,
    50.0,
    55.0,
    60.0,
    65.0,
    70.0,
    75.0,
    80.0,
    85.0,
    90.0,
    95.0,
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
    # golden = xr.open_dataset(EXAMPLE_BT_NC)
    example_ds = xr.open_dataset(PACKAGE_DIR / '..' / 'NH_20200101_py_NRT_amsr2.nc')
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
        add_labels=False,
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


def _get_valid_icemask():
    ds = xr.open_dataset(
        '/projects/DATASETS/nsidc0622_valid_seaice_masks'
        '/NIC_valid_ice_mask.N25km.01.1972-2007.nc'
    )

    return ds


if __name__ == '__main__':
    # Get and save an image of the NH example data produced by our python code.
    example_ds = get_example_output()
    save_n_conc_image(example_ds.conc, Path('/tmp/NH_20200101_py_amsr2.png'))

    # Do the same for the bootstrap concentration field that comes with the
    # AU_SI25 data.
    au_si25_conc = get_au_si25_bt_conc()
    save_n_conc_image(au_si25_conc, Path('/tmp/NH_20200101_au_si25_amsr2.png'))

    # Do a difference between the two images.
    aui_si25_conc_masked = au_si25_conc.where(au_si25_conc != 110, 0)

    # Mask out lakes (value of 4)
    valid_icemask = _get_valid_icemask()
    aui_si25_conc_masked = aui_si25_conc_masked.where(
        valid_icemask.valid_ice_flag.data != 4,
        0,
    )

    # mask out pole hole
    pole_hole_path = (
        PACKAGE_DIR
        / '../legacy/SB2_NRT_programs'
        / '../SB2_NRT_programs/ANCILLARY/np_holemask.ssmi_f17'
    ).resolve()
    holemask = (
        np.fromfile(pole_hole_path, dtype=np.int16).reshape(448, 304).astype(bool)
    )
    aui_si25_conc_masked = aui_si25_conc_masked.where(~holemask, 110)

    diff = example_ds.conc - aui_si25_conc_masked
    plt.clf()
    plt.cla()
    plt.close()
    ax = plt.axes((0, 0, 1, 1), projection=N_PROJ)
    diff.plot.imshow(
        ax=ax,
        add_colorbar=True,
        transform=N_PROJ,
        # (left, right, bottom, top)
        extent=[-3850000.000, 3750000.0, -5350000.0, 5850000.000],
    )
    ax.coastlines()
    plt.savefig(
        '/tmp/NH_20200101_diff_amsr2.png',
        bbox_inches='tight',
        pad_inches=0.05,
    )

    # Histogram
    plt.clf()
    plt.cla()
    plt.close()

    diff = diff.data.flatten()
    diff_excluding_0 = diff[diff != 0]
    breakpoint()
    plt.hist(
        diff_excluding_0,
        bins=list(range(-100, 120, 5)),
        log=True,
    )

    plt.xticks(list(range(-100, 120, 20)))

    plt.savefig(
        '/tmp/NH_20200101_diff_hist_amsr2.png',
        bbox_inches='tight',
        pad_inches=0.05,
    )
