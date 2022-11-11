import datetime as dt
from itertools import product
from pathlib import Path
from typing import get_args

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr
from loguru import logger

from cdr_amsr2._types import Hemisphere
from cdr_amsr2.compare.ref_data import cdr_for_date_range
from cdr_amsr2.constants import CDR_DATA_DIR
from cdr_amsr2.fetch.au_si import AU_SI_RESOLUTIONS
from cdr_amsr2.util import date_range, standard_output_filename

OUTPUT_DIR = Path('/tmp/compare_cdr/')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def amsr2_cdr_for_date_range(
    *,
    start_date: dt.date,
    end_date: dt.date,
    resolution: AU_SI_RESOLUTIONS,
    hemisphere: Hemisphere,
) -> xr.Dataset:
    """Return a xarray Dataset with CDR `conc` var indexed by date."""
    conc_datasets = []
    conc_dates = []
    # TODO: could use `xr.open_mfdataset`, especially if we setup `conc` with a
    # time dimension.
    for date in date_range(start_date=start_date, end_date=end_date):
        expected_fn = standard_output_filename(
            hemisphere=hemisphere,
            date=date,
            sat='u2',
            resolution=resolution,
            algorithm='cdr',
        )
        expected_path = CDR_DATA_DIR / expected_fn
        if not expected_path.is_file():
            raise FileNotFoundError(f'Unexpectedly missing file: {expected_path}')

        ds = xr.open_dataset(expected_path)
        conc_datasets.append(ds)
        conc_dates.append(date)

    merged = xr.concat(
        conc_datasets,
        pd.DatetimeIndex(conc_dates, name='date'),
    )

    return merged


def extent_from_conc(
    *, conc: xr.DataArray, area_grid: npt.NDArray, extent_threshold=15
) -> xr.DataArray:
    """Return extents in mkm2."""
    has_ice = (conc >= extent_threshold) & (conc <= 100)
    extents = (has_ice.astype(int) * area_grid).sum(dim=('y', 'x'))
    # convert to millions of km2.
    extents = extents / 1_000_000
    extents.name = 'extent'  # noqa

    return extents


def area_from_conc(
    *, conc: xr.DataArray, area_grid: npt.NDArray, area_threshold=15
) -> xr.DataArray:
    """Return areas in mkm2."""
    has_ice = (conc >= area_threshold) & (conc <= 100)
    conc = conc.where(has_ice, other=0)
    areas = ((conc / 100) * area_grid).sum(dim=('y', 'x'))
    # convert to millions of km2.
    areas = areas / 1_000_000
    areas.name = 'area'  # noqa

    return areas


def _get_ps_area_grid(
    *, hemisphere: Hemisphere, resolution: AU_SI_RESOLUTIONS
) -> npt.NDArray:
    """Return the area grid for the given hemisphere and resolution.

    Units are km2.
    """
    data_dir = Path('/projects/DATASETS/nsidc0771_polarstereo_anc_grid_info/')

    # TODO: we refer to the 12.5km data as '12' in most parts of the code. These
    # filenames have '12.5'.
    if resolution == '12':
        resolution = '12.5'  # type: ignore[assignment]

    ds = xr.open_dataset(
        data_dir
        / f'NSIDC0771_CellArea_PS_{hemisphere[0].upper()}{resolution}km_v1.0.nc'
    )

    area_grid = ds.cell_area.data

    # Grid areas are in m2. Convert to km2
    area_grid = area_grid / 1_000_000

    ds.close()

    return area_grid


def compare_timeseries(
    *,
    kind,
    hemisphere: Hemisphere,
    start_date: dt.date,
    end_date: dt.date,
    resolution: AU_SI_RESOLUTIONS,
):

    amsr2_cdr = amsr2_cdr_for_date_range(
        start_date=start_date,
        end_date=end_date,
        resolution=resolution,
        hemisphere=hemisphere,
    )
    logger.info('Obtained AMSR2 CDR')

    cdr = cdr_for_date_range(
        start_date=start_date,
        end_date=end_date,
        hemisphere=hemisphere,
        resolution=resolution,
    )
    logger.info('Obtained CDR')

    area_grid = _get_ps_area_grid(hemisphere=hemisphere, resolution=resolution)
    logger.info('Obtained area grid')

    if kind == 'extent':
        amsr2_cdr_timeseries = extent_from_conc(
            conc=amsr2_cdr.conc,
            area_grid=area_grid,
        )
        cdr_timeseries = extent_from_conc(
            conc=cdr.conc,
            area_grid=area_grid,
        )
    elif kind == 'area':
        amsr2_cdr_timeseries = area_from_conc(
            conc=amsr2_cdr.conc,
            area_grid=area_grid,
        )
        cdr_timeseries = area_from_conc(
            conc=cdr.conc,
            area_grid=area_grid,
        )

    else:
        raise NotImplementedError('')

    logger.info('Building plots.')
    fig, ax = plt.subplots(
        nrows=2, ncols=1, subplot_kw={'aspect': 'auto', 'autoscale_on': True}
    )
    logger.info('Made subplots.')

    _ax = ax[0]

    _ax.plot(
        amsr2_cdr_timeseries.date,
        amsr2_cdr_timeseries.data,
        label=f'AMSR2 (AU_SI{resolution}) CDR',
    )
    logger.info('subplot plot 1')
    _ax.plot(cdr_timeseries.date, cdr_timeseries.data, label='CDR')
    logger.info('subplot plot 2')
    max_value = np.max([cdr_timeseries.max(), amsr2_cdr_timeseries.max()])
    logger.info(f'Got max value: {max_value}')
    _ax.set(
        xlabel='date',
        ylabel=f'{kind.capitalize()} (Millions of square kilometers)',
        title=kind.capitalize(),
        xlim=(cdr_timeseries.date.min(), cdr_timeseries.date.max()),
        yticks=np.arange(0, float(max_value) + 2, 2),
    )
    logger.info('ax set')
    _ax.legend()
    logger.info('legend set')
    _ax.grid()
    logger.info('Built plot 0')

    _ax = ax[1]
    diff = amsr2_cdr_timeseries - cdr_timeseries
    _ax.plot(diff.date, diff.data)
    _ax.set(
        xlabel='date',
        ylabel=f'{kind.capitalize()} (Millions of square kilometers)',
        title=f'AMSR2 CDR minus CDR {kind}',
        xlim=(diff.date.min(), diff.date.max()),
    )
    _ax.grid()
    logger.info('Built plot 1')

    fig.set_size_inches(w=25, h=16)
    fig.suptitle(f'{hemisphere} {kind}')
    out_fn = (
        f'{hemisphere}_{resolution}km'
        f'_{start_date:%Y%m%d}_{end_date:%Y%m%d}_{kind}_comparison.png'
    )
    out_fp = OUTPUT_DIR / out_fn
    fig.savefig(
        out_fp,
        bbox_inches='tight',
        pad_inches=0.05,
    )
    logger.info('Created output - done!')

    plt.clf()


if __name__ == '__main__':
    start_date = dt.date(2021, 1, 1)
    end_date = dt.date(2021, 12, 31)

    for hemisphere, resolution in product(
        get_args(Hemisphere), get_args(AU_SI_RESOLUTIONS)
    ):
        if resolution == '25':
            continue
        compare_timeseries(
            kind='extent',
            hemisphere=hemisphere,
            start_date=start_date,
            end_date=end_date,
            resolution=resolution,
        )
        compare_timeseries(
            kind='area',
            hemisphere=hemisphere,
            start_date=start_date,
            end_date=end_date,
            resolution=resolution,
        )
