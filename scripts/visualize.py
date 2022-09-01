"""One-off script to produce visualizations for the NOAA stakeholders meeting.

Meeting took place on June 30th, 2022 and went well!

This code is a bit of a mess, but may serve as reference for future
visualization code.
"""
import datetime as dt
from pathlib import Path

import numpy as np
import numpy.typing as npt
import xarray as xr
from matplotlib import pyplot as plt

import cdr_amsr2.nt.compute_nt_ic as nt
from cdr_amsr2._types import Hemisphere
from cdr_amsr2.bt.api import amsr2_bootstrap
from cdr_amsr2.bt.masks import get_ps_invalid_ice_mask
from cdr_amsr2.fetch import au_si
from cdr_amsr2.masks import get_ps_pole_hole_mask
from cdr_amsr2.nt.api import amsr2_nasateam, original_example
from cdr_amsr2.nt.masks import get_ps25_sst_mask
from cdr_amsr2.util import get_ps25_grid_shape

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


def _flip_and_scale(input_conc_ds):
    flipped_and_scaled = input_conc_ds.copy()
    # flip the image to be 'right-side' up
    flipped_and_scaled = flipped_and_scaled.reindex(
        y=input_conc_ds.y[::-1], x=input_conc_ds.x
    )

    # scale the data by 10 and convert to int
    flipped_and_scaled['conc'] = flipped_and_scaled.conc / 10

    # Round the data as integers.
    flipped_and_scaled['conc'] = (flipped_and_scaled.conc + 0.5).astype(np.uint8)

    return flipped_and_scaled


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
    example_ds = _flip_and_scale(example_ds)

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


def get_au_si_bt_nt_conc(
    *,
    date: dt.date,
    hemisphere: Hemisphere,
    resolution: au_si.AU_SI_RESOLUTIONS,
) -> tuple[xr.DataArray, xr.DataArray]:
    ds = au_si._get_au_si_data_fields(
        # TODO: DRY out base dir defualt. No need to pass this around...
        base_dir=Path(f'/ecs/DP1/AMSA/AU_SI{resolution}.001/'),
        date=date,
        hemisphere=hemisphere,
        resolution=resolution,  # type: ignore[arg-type]
    )

    # flip the image to be 'right-side' up
    ds = ds.reindex(YDim=ds.YDim[::-1], XDim=ds.XDim)
    ds = ds.rename({'YDim': 'y', 'XDim': 'x'})

    nt_conc = getattr(ds, f'SI_{resolution}km_{hemisphere[0].upper()}H_ICECON_DAY')
    diff = getattr(ds, f'SI_{resolution}km_{hemisphere[0].upper()}H_ICEDIFF_DAY')
    bt_conc = nt_conc + diff

    return (bt_conc, nt_conc)


def _mask_data(
    data,
    hemisphere: Hemisphere,
    date: dt.date,
    invalid_icemask,
    pole_hole_mask=None,
):
    aui_si25_conc_masked = data.where(data != 110, 0)

    # Mask out invalid ice (the AU_SI products have conc values in lakes. We
    # don't include those in our valid ice masks.
    aui_si25_conc_masked = aui_si25_conc_masked.where(cond=~invalid_icemask, other=0)

    if hemisphere == 'north' and pole_hole_mask is not None:
        aui_si25_conc_masked = aui_si25_conc_masked.where(
            cond=~pole_hole_mask, other=110
        )

    return aui_si25_conc_masked


def do_comparisons(
    *,
    # concentration field produced by our code
    cdr_amsr2_conc: xr.DataArray,
    # concentration against which the cdr_amsr2_conc will be compared.
    comparison_conc: xr.DataArray,
    hemisphere: Hemisphere,
    invalid_icemask: npt.NDArray[np.bool_],
    date: dt.date,
    # e.g., `AU_SI25`
    product_name: str,
    pole_hole_mask: npt.NDArray[np.bool_] | None = None,
) -> None:
    """Create figure showing comparison between concentration fields."""
    fig, ax = plt.subplots(
        nrows=2, ncols=2, subplot_kw={'aspect': 'auto', 'autoscale_on': True}
    )

    # Get the bootstrap concentration field that comes with the
    # AU_SI data.
    _ax = ax[0][0]
    _ax.title.set_text(f'{product_name} provided conc')
    _ax.set_xticks([])
    _ax.set_yticks([])
    save_conc_image(
        conc_array=comparison_conc,
        hemisphere=hemisphere,
        ax=_ax,
    )

    _ax = ax[0][1]
    _ax.title.set_text('Python calculated conc')
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

    diff = cdr_amsr2_conc - comparison_conc_masked
    _ax = ax[1][0]
    _ax.title.set_text(f'Python minus {product_name} conc')
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

    fig.suptitle(f'{product_name} {hemisphere[0].upper()}H {date:%Y-%m-%d}')
    fig.set_size_inches(w=20, h=16)
    fig.savefig(
        OUTPUT_DIR / f'{product_name}_{hemisphere[0].upper()}H_{date:%Y-%m-%d}.png',
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
    au_si25_conc, _ = get_au_si_bt_nt_conc(
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
        product_name=f'AU_SI{resolution}',
        pole_hole_mask=holemask,
    )


def _fix_conc_field_for_nt(ds):
    # land == 25. Concentrations > 100 exist. # pole hole/missing == 252. We'll
    # need to 'fix' this so that visualizations come out looking right
    # (colorbar)
    new_ds = ds.copy()

    # Account for concentrations > 100.
    # TODO: this logic should probably be moved to the nasateam alg.
    new_ds['conc'] = xr.where(
        (new_ds.conc > 100) & (new_ds.conc < 200), 100, new_ds.conc
    )

    # Make the nt output land value the expected land value
    # TODO: how is 25 land and not a valid conc value?
    new_ds['conc'] = xr.where(new_ds.conc == 25, 120, new_ds.conc)

    # Make the missing areas the expected missing value (110)
    new_ds['conc'] = xr.where(new_ds.conc == 252, 110, new_ds.conc)

    return new_ds


def do_comparison_original_example_nt(*, hemisphere: Hemisphere):  # noqa
    """Compare original examples from Goddard for nasateam."""
    # TODO: our api for nasateam and bootstrap should return consistent fields
    # (same pole hole / missing value, 'right-side' up, etc.

    def _read_goddard_nasateam_file(filename: Path, /):
        with open(filename, 'rb') as fp:
            fp.read(300)
            data = np.fromfile(fp, dtype='>i2')
        return data

    our_conc_ds = _flip_and_scale(original_example(hemisphere=hemisphere))
    our_conc_ds = _fix_conc_field_for_nt(our_conc_ds)
    regression_conc_ds = _flip_and_scale(
        xr.Dataset(
            {
                'conc': (
                    ('y', 'x'),
                    _read_goddard_nasateam_file(
                        (
                            Path('/share/apps/amsr2-cdr/cdr_testdata')
                            / 'nt_f17_regression'
                            / f'{hemisphere[0]}ssss1d17tcon2018001.spill_sst'
                        )
                    ).reshape(get_ps25_grid_shape(hemisphere=hemisphere)),
                )
            }
        )
    )
    regression_conc_ds = _fix_conc_field_for_nt(regression_conc_ds)

    date = dt.date(2018, 1, 1)
    invalid_icemask = get_ps25_sst_mask(hemisphere=hemisphere, date=date)
    do_comparisons(
        cdr_amsr2_conc=our_conc_ds.conc,
        comparison_conc=regression_conc_ds.conc,
        hemisphere=hemisphere,
        invalid_icemask=invalid_icemask,
        date=date,
        product_name='f17_final_25km',
        pole_hole_mask=nt._get_polehole_mask() if hemisphere == 'north' else None,
    )


def do_comparison_amsr2_nt(*, hemisphere='north'):
    date = dt.date(2022, 8, 1)
    resolution = '25'

    our_conc_ds = amsr2_nasateam(
        date=date,
        hemisphere=hemisphere,
    )
    our_conc_ds = _flip_and_scale(our_conc_ds)
    our_conc_ds = _fix_conc_field_for_nt(our_conc_ds)

    _, comparison_conc = get_au_si_bt_nt_conc(
        date=date,
        hemisphere=hemisphere,
        resolution=resolution,  # type: ignore[arg-type]
    )

    do_comparisons(
        cdr_amsr2_conc=our_conc_ds.conc,
        comparison_conc=comparison_conc,
        hemisphere=hemisphere,
        invalid_icemask=get_ps25_sst_mask(hemisphere=hemisphere, date=date),
        date=date,
        product_name='AU_SI25',
        pole_hole_mask=nt._get_polehole_mask() if hemisphere == 'north' else None,
    )


if __name__ == '__main__':
    # do_comparisons_au_si_bt(
    #     hemisphere='north',
    #     date=dt.date(2022, 8, 1),
    #     resolution='12',
    # )

    # do_comparison_original_example_nt(hemisphere='south')

    do_comparison_amsr2_nt(hemisphere='north')
