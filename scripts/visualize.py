"""One-off script to produce visualizations for the NOAA stakeholders meeting.

Meeting took place on June 30th, 2022 and went well!

This code is a bit of a mess, but may serve as reference for future
visualization code.
"""
import datetime as dt
from pathlib import Path

import numpy as np
import xarray as xr
from matplotlib import pyplot as plt

from cdr_amsr2._types import Hemisphere
from cdr_amsr2.bt.api import amsr2_bootstrap
from cdr_amsr2.constants import PACKAGE_DIR
from cdr_amsr2.fetch import au_si

OUTPUT_DIR = Path('/tmp/diffs/')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


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


def get_example_output(*, hemisphere: Hemisphere, date: dt.date) -> xr.Dataset:
    """Get the example AMSR2 output from our python code.

    * Flip the data so that North is 'up'.
    * Scale the data by 10 and round to np.uint8 dtype.
    """
    example_ds = amsr2_bootstrap(
        date=date,
        hemisphere=hemisphere,
        # TODO: parameterize this.
        resolution='25',
    )
    # flip the image to be 'right-side' up
    example_ds = example_ds.reindex(y=example_ds.y[::-1], x=example_ds.x)

    # scale the data by 10 and convert to int
    example_ds['conc'] = example_ds.conc / 10

    # Round the data as integers.
    example_ds['conc'] = (example_ds.conc + 0.5).astype(np.uint8)

    return example_ds


def save_conc_image(*, conc_array: xr.DataArray, hemisphere: Hemisphere, ax) -> None:
    """Create an image representing the conc field."""
    conc_array.plot.imshow(  # type: ignore[attr-defined]
        ax=ax,
        colors=COLORS,
        levels=COLORBOUNDS,
        add_colorbar=False,
        add_labels=False,
    )


def get_au_si25_bt_conc(*, date: dt.date, hemisphere: Hemisphere) -> xr.DataArray:
    ds = au_si._get_au_si_data_fields(
        base_dir=Path('/ecs/DP1/AMSA/AU_SI25.001/'),
        date=date,
        hemisphere=hemisphere,
        resolution='25',
    )

    # flip the image to be 'right-side' up
    ds = ds.reindex(YDim=ds.YDim[::-1], XDim=ds.XDim)
    ds = ds.rename({'YDim': 'y', 'XDim': 'x'})

    nt_conc = getattr(ds, f'SI_25km_{hemisphere[0].upper()}H_ICECON_DAY')
    diff = getattr(ds, f'SI_25km_{hemisphere[0].upper()}H_ICEDIFF_DAY')
    bt_conc = nt_conc + diff

    return bt_conc


def _get_valid_icemask():
    ds = xr.open_dataset(
        '/projects/DATASETS/nsidc0622_valid_seaice_masks'
        '/NIC_valid_ice_mask.N25km.01.1972-2007.nc'
    )

    return ds


def _mask_data(data, hemisphere: Hemisphere):
    aui_si25_conc_masked = data.where(data != 110, 0)

    if hemisphere == 'north':
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

    return aui_si25_conc_masked


def do_comparisons_ausi25(*, hemisphere: Hemisphere, date: dt.date) -> None:
    # Get and save an image of the example data produced by our python code.
    example_ds = get_example_output(hemisphere=hemisphere, date=date)
    fig, ax = plt.subplots(nrows=2, ncols=2)
    _ax = ax[0][0]
    _ax.title.set_text('Python calculated conc')
    _ax.set_xticks([])
    _ax.set_yticks([])
    save_conc_image(
        conc_array=example_ds.conc,
        hemisphere=hemisphere,
        ax=_ax,
    )

    # Do the same for the bootstrap concentration field that comes with the
    # AU_SI25 data.
    au_si25_conc = get_au_si25_bt_conc(date=date, hemisphere=hemisphere)
    _ax = ax[0][1]
    _ax.title.set_text('AU_SI25 provided conc')
    _ax.set_xticks([])
    _ax.set_yticks([])
    save_conc_image(
        conc_array=au_si25_conc,
        hemisphere=hemisphere,
        ax=_ax,
    )

    # Do a difference between the two images.
    aui_si25_conc_masked = _mask_data(au_si25_conc, hemisphere)

    diff = example_ds.conc - aui_si25_conc_masked
    _ax = ax[1][0]
    _ax.title.set_text('Python minus AU_SI25 conc')
    _ax.set_xticks([])
    _ax.set_yticks([])
    diff.plot.imshow(
        ax=_ax,
        add_colorbar=True,
        add_labels=False,
    )

    # Histogram
    diff = diff.data.flatten()
    diff_excluding_0 = diff[diff != 0]

    _ax = ax[1][1]
    _ax.title.set_text('Histogram of non-zero differences')
    _ax.hist(
        diff_excluding_0,
        bins=list(range(-100, 120, 5)),
        log=True,
    )

    plt.xticks(list(range(-100, 120, 20)))

    fig.suptitle(f'{hemisphere[0].upper()}H {date:%Y-%m-%d}')
    fig.set_size_inches(w=20, h=16)
    fig.savefig(
        OUTPUT_DIR / f'{hemisphere[0].upper()}H_{date:%Y-%m-%d}.png',
        bbox_inches='tight',
        pad_inches=0.05,
    )


if __name__ == '__main__':
    do_comparisons_ausi25(hemisphere='south', date=dt.date(2022, 8, 1))
