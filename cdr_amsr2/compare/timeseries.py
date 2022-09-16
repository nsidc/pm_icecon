import datetime as dt
from functools import cache
from pathlib import Path
from typing import get_args

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr
from seaice.nasateam.area_grids import NORTH_AREA_GRID, SOUTH_AREA_GRID

from cdr_amsr2._types import Hemisphere
from cdr_amsr2.compare.ref_data import cdr_for_date_range
from cdr_amsr2.fetch.au_si import AU_SI_RESOLUTIONS
from cdr_amsr2.util import date_range, standard_output_filename

CDR_DATA_DIR = Path('/share/apps/amsr2-cdr/cdr_data')
OUTPUT_DIR = Path('/tmp/compare_cdr/')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@cache
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
    """Returns extents in mkm2"""
    has_ice = (conc >= extent_threshold) & (conc <= 100)
    extents = (has_ice.astype(int) * area_grid).sum(dim=('y', 'x'))
    extents = extents / 1_000_000
    extents.name = 'extent'

    return extents


def area_from_conc(
    *, conc: xr.DataArray, area_grid: npt.NDArray, area_threshold=15
) -> xr.DataArray:
    """Returns areas in mkm2"""
    has_ice = (conc >= area_threshold) & (conc <= 100)
    conc = conc.where(has_ice, other=0)
    areas = ((conc / 100) * area_grid).sum(dim=('y', 'x'))
    areas = areas / 1_000_000
    areas.name = 'area'

    return areas


def compare_timeseries(*, kind, hemisphere: Hemisphere):
    start_date = dt.date(2021, 1, 1)
    end_date = dt.date(2021, 12, 31)
    resolution = '25'

    amsr2_cdr = amsr2_cdr_for_date_range(
        start_date=start_date,
        end_date=end_date,
        resolution='25km',
        hemisphere=hemisphere,
    )

    cdr = cdr_for_date_range(
        start_date=start_date,
        end_date=end_date,
        hemisphere=hemisphere,
        resolution=resolution,
    )

    if kind == 'extent':
        amsr2_cdr_timeseries = extent_from_conc(
            conc=amsr2_cdr.conc,
            area_grid=NORTH_AREA_GRID if hemisphere == 'north' else SOUTH_AREA_GRID,
        )
        cdr_timeseries = extent_from_conc(
            conc=cdr.conc,
            area_grid=NORTH_AREA_GRID if hemisphere == 'north' else SOUTH_AREA_GRID,
        )
    elif kind == 'area':
        amsr2_cdr_timeseries = area_from_conc(
            conc=amsr2_cdr.conc,
            area_grid=NORTH_AREA_GRID if hemisphere == 'north' else SOUTH_AREA_GRID,
        )
        cdr_timeseries = area_from_conc(
            conc=cdr.conc,
            area_grid=NORTH_AREA_GRID if hemisphere == 'north' else SOUTH_AREA_GRID,
        )

    else:
        raise NotImplementedError('')

    fig, ax = plt.subplots(
        nrows=2, ncols=1, subplot_kw={'aspect': 'auto', 'autoscale_on': True}
    )

    _ax = ax[0]

    _ax.plot(
        amsr2_cdr_timeseries.date,
        amsr2_cdr_timeseries.data,
        label=f'AMSR2 (AU_SI{resolution}) CDR',
    )
    _ax.plot(cdr_timeseries.date, cdr_timeseries.data, label='CDR')
    max_value = np.max([cdr_timeseries.max(), amsr2_cdr_timeseries.max()])
    _ax.set(
        xlabel='date',
        ylabel=f'{kind.capitalize()} (Millions of square kilometers)',
        title=kind.capitalize(),
        xlim=(cdr_timeseries.date.min(), cdr_timeseries.date.max()),
        yticks=np.arange(0, float(max_value) + 2, 2),
    )
    _ax.legend()
    _ax.grid()

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

    fig.set_size_inches(w=25, h=16)
    fig.suptitle(f'{hemisphere} {kind}')
    fig.savefig(
        OUTPUT_DIR
        / f'{hemisphere}_{start_date:%Y%m%d}_{end_date:%Y%m%d}_{kind}_comparison.png',
        bbox_inches='tight',
        pad_inches=0.05,
    )

    plt.clf()


if __name__ == '__main__':
    for hemisphere in get_args(Hemisphere):
        compare_timeseries(kind='extent', hemisphere=hemisphere)
        compare_timeseries(kind='area', hemisphere=hemisphere)
