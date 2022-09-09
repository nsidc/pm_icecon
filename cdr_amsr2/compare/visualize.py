"""One-off script to produce visualizations for the NOAA stakeholders meeting.

Meeting took place on June 30th, 2022 and went well!

This code is a bit of a mess, but may serve as reference for future
visualization code.
"""
import datetime as dt
from pathlib import Path
from typing import get_args

import numpy as np
import numpy.typing as npt
import xarray as xr
from matplotlib import pyplot as plt

import cdr_amsr2.nt.api as nt_api
from cdr_amsr2._types import Hemisphere
from cdr_amsr2.bt.api import amsr2_bootstrap
from cdr_amsr2.bt.masks import get_ps_invalid_ice_mask
from cdr_amsr2.compare.ref_data import get_au_si_bt_conc, get_sea_ice_index
from cdr_amsr2.constants import DEFAULT_FLAG_VALUES
from cdr_amsr2.fetch import au_si
from cdr_amsr2.masks import get_ps_pole_hole_mask
from cdr_amsr2.nt.masks import get_ps25_sst_mask

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
    '#777777',  # 254land
    '#e9cb00',  # 255missing
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
    254.001,
    255.001,
]


def _flip(input_conc_ds):
    flipped = input_conc_ds.copy()
    # flip the image to be 'right-side' up
    flipped = flipped.reindex(
        y=input_conc_ds.y[::-1], x=input_conc_ds.x
    )

    return flipped


# TODO: rename this func.
def get_example_output(
    *, hemisphere: Hemisphere, date: dt.date, resolution: au_si.AU_SI_RESOLUTIONS
) -> xr.Dataset:
    """Get the example AMSR2 output from our python code.

    * Flip the data so that North is 'up'.
    * Scale the data by 10 and round to np.uint8 dtype.
    """
    example_ds = amsr2_bootstrap(
        date=date,
        hemisphere=hemisphere,
        resolution=resolution,  # type: ignore[arg-type]
    )
    example_ds = _flip(example_ds)

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


def _mask_data(
    data,
    hemisphere: Hemisphere,
    date: dt.date,
    invalid_icemask,
    pole_hole_mask=None,
):
    masked = data.copy()

    # Mask out invalid ice (the AU_SI products have conc values in lakes. We
    # don't include those in our valid ice masks.
    masked = masked.where(cond=~invalid_icemask, other=0)

    if hemisphere == 'north' and pole_hole_mask is not None:
        masked = masked.where(cond=~pole_hole_mask, other=0)

    return masked


def do_comparisons(
    *,
    # concentration field produced by our code
    cdr_amsr2_conc: xr.DataArray,
    # e.g., `AU_SI25`
    cdr_amsr2_dataproduct: str,
    cdr_amsr2_algorithm: str,
    # concentration against which the cdr_amsr2_conc will be compared.
    comparison_conc: xr.DataArray,
    # E.g., 'SII'
    comparison_dataproduct: str,
    hemisphere: Hemisphere,
    invalid_icemask: npt.NDArray[np.bool_],
    date: dt.date,
    pole_hole_mask: npt.NDArray[np.bool_] | None = None,
) -> None:
    """Create figure showing comparison between concentration fields."""
    fig, ax = plt.subplots(
        nrows=2, ncols=2, subplot_kw={'aspect': 'auto', 'autoscale_on': True}
    )

    # Visualize the comparison conc.
    _ax = ax[0][0]
    _ax.title.set_text(f'{comparison_dataproduct} provided conc')
    _ax.set_xticks([])
    _ax.set_yticks([])
    save_conc_image(
        conc_array=comparison_conc,
        hemisphere=hemisphere,
        ax=_ax,
    )

    _ax = ax[0][1]
    _ax.title.set_text(
        f'Python calculated conc from {cdr_amsr2_dataproduct}'
        f' using the {cdr_amsr2_algorithm} algorithm.'
    )
    _ax.set_xticks([])
    _ax.set_yticks([])
    save_conc_image(
        conc_array=cdr_amsr2_conc,
        hemisphere=hemisphere,
        ax=_ax,
    )

    # Do a difference between the two images.
    comparison_conc_masked = _mask_data(
        comparison_conc,
        hemisphere,
        date,
        invalid_icemask,
        pole_hole_mask=pole_hole_mask,
    )

    # Exclude areas that are not valid concentrations in both fields (exlude
    # mismatches between land masks)
    cdr_amsr2_conc_validice = (cdr_amsr2_conc >= 0) & (cdr_amsr2_conc <= 100)
    # fmt: off
    comparison_conc_validice = (
        (comparison_conc_masked >= 0)
        & (comparison_conc_masked <= 100)
    )
    # fmt: on
    common_validice = cdr_amsr2_conc_validice & comparison_conc_validice
    cdr_amsr2_conc = cdr_amsr2_conc.where(common_validice, 0)
    comparison_conc_masked = comparison_conc_masked.where(common_validice, 0)
    diff = cdr_amsr2_conc - comparison_conc_masked
    _ax = ax[1][0]
    _ax.title.set_text('Python minus comparison conc')
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

    pixels_different = len(diff_excluding_0)
    # TODO: filter total_pixels to just those that could be valid ice (not
    # masked by land/invalid ice mask)
    total_pixels = len(diff)
    percent_different = (pixels_different / total_pixels) * 100

    _ax = ax[1][1]
    _ax.title.set_text(
        'Histogram of non-zero differences'
        '\n'
        f'{percent_different:.3}% of pixels are different.'
        '\n'
        f'Min difference: {diff_excluding_0.min():.6}.'
        '\n'
        f'Max difference: {diff_excluding_0.max():.6}.'
    )
    _ax.hist(
        diff_excluding_0,
        bins=list(range(-100, 120, 5)),
        log=True,
    )

    plt.xticks(list(range(-100, 120, 20)))

    fig.suptitle(
        f'{cdr_amsr2_dataproduct} vs {comparison_dataproduct}'
        f' {hemisphere[0].upper()}H {date:%Y-%m-%d}'
    )
    fig.set_size_inches(w=20, h=16)
    fig.savefig(
        (
            OUTPUT_DIR
            / (
                f'{cdr_amsr2_dataproduct}_vs_{comparison_dataproduct}'
                f'_{hemisphere[0].upper()}H_{date:%Y-%m-%d}.png'
            )
        ),
        bbox_inches='tight',
        pad_inches=0.05,
    )


def do_comparisons_au_si_bt(  # noqa
    *,
    hemisphere: Hemisphere,
    date: dt.date,
    resolution: au_si.AU_SI_RESOLUTIONS,
) -> None:
    """Create figure showing comparison for AU_SI{25|12}."""
    au_si25_conc = get_au_si_bt_conc(
        date=date, hemisphere=hemisphere, resolution=resolution
    )

    # Get the example data produced by our python code.
    example_ds = get_example_output(
        hemisphere=hemisphere, date=date, resolution=resolution
    )

    # TODO: better to exclude lakes explicitly via the land mask?
    # True areas are invalid ice. False areas are possibly valid (includes land)
    invalid_icemask = get_ps_invalid_ice_mask(
        hemisphere=hemisphere,
        date=date,
        resolution=resolution,
    )

    if hemisphere == 'north':
        holemask = get_ps_pole_hole_mask(resolution=resolution)
    else:
        holemask = None

    do_comparisons(
        cdr_amsr2_conc=example_ds.conc,
        comparison_conc=au_si25_conc,
        hemisphere=hemisphere,
        invalid_icemask=invalid_icemask,
        date=date,
        cdr_amsr2_dataproduct=f'AU_SI{resolution}',
        cdr_amsr2_algorithm='bootstrap',
        comparison_dataproduct=f'AU_SI{resolution}',
        pole_hole_mask=holemask,
    )


def _fix_nt_outputs(conc_ds):
    conc_ds = _flip(conc_ds)
    conc_ds['conc'] = xr.where(
        (conc_ds.conc > 100) & (conc_ds.conc < 200), 100, conc_ds.conc
    )

    return conc_ds


def compare_original_nt_to_sii(*, hemisphere: Hemisphere) -> None:  # noqa
    """Compare original examples from Goddard for nasateam."""
    our_conc_ds = _fix_nt_outputs(nt_api.original_example(hemisphere=hemisphere))

    date = dt.date(2018, 1, 1)
    sii_conc_ds = get_sea_ice_index(hemisphere=hemisphere, date=date)

    do_comparisons(
        cdr_amsr2_conc=our_conc_ds.conc,
        comparison_conc=sii_conc_ds.conc,
        hemisphere=hemisphere,
        invalid_icemask=get_ps25_sst_mask(hemisphere=hemisphere, date=date),
        date=date,
        cdr_amsr2_dataproduct='goddard_example_f17',
        cdr_amsr2_algorithm='nasateam',
        comparison_dataproduct='SII_25km',
        pole_hole_mask=get_ps_pole_hole_mask(resolution='25')
        if hemisphere == 'north'
        else None,
    )


def compare_amsr_nt_to_sii(  # noqa
    *, hemisphere: Hemisphere, resolution: au_si.AU_SI_RESOLUTIONS
) -> None:
    date = dt.date(2022, 8, 1)
    # date = dt.date(2018, 1, 1)

    sii_conc_ds = get_sea_ice_index(
        hemisphere=hemisphere, date=date, resolution=resolution
    )
    our_conc_ds = _fix_nt_outputs(
        nt_api.amsr2_nasateam(
            date=date,
            hemisphere=hemisphere,
            resolution=resolution,
        )
    )

    do_comparisons(
        cdr_amsr2_conc=our_conc_ds.conc,
        comparison_conc=sii_conc_ds.conc,
        hemisphere=hemisphere,
        invalid_icemask=get_ps_invalid_ice_mask(
            hemisphere=hemisphere, date=date, resolution=resolution
        ),
        date=date,
        cdr_amsr2_dataproduct=f'AU_SI{resolution}',
        cdr_amsr2_algorithm='nasateam',
        comparison_dataproduct=f'SII_{resolution}km',
        pole_hole_mask=get_ps_pole_hole_mask(resolution=resolution)
        if hemisphere == 'north'
        else None,
    )


if __name__ == '__main__':
    for hemisphere in get_args(Hemisphere):
        do_comparisons_au_si_bt(
            hemisphere=hemisphere,
            date=dt.date(2022, 8, 1),
            resolution='12',
        )
    #     # compare_original_nt_to_sii(hemisphere=hemisphere)
    #     compare_amsr_nt_to_sii(
    #         hemisphere=hemisphere,  # type: ignore[arg-type]
    #         resolution='12',  # type: ignore[arg-type]
    #     )
