import datetime as dt
from pathlib import Path

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


if __name__ == '__main__':
    hemisphere = 'north'
    start_date = dt.date(2021, 1, 1)
    end_date = dt.date(2021, 12, 31)
    resolution = '25'

    amsr2_cdr = amsr2_cdr_for_date_range(
        start_date=start_date,
        end_date=end_date,
        resolution='25km',
        hemisphere=hemisphere,
    )

    amsr2_cdr_extent = extent_from_conc(
        conc=amsr2_cdr.conc,
        area_grid=NORTH_AREA_GRID if hemisphere == 'north' else SOUTH_AREA_GRID,
    )

    cdr = cdr_for_date_range(
        start_date=start_date,
        end_date=end_date,
        hemisphere=hemisphere,
        resolution=resolution,
    )

    cdr_extent = extent_from_conc(
        conc=cdr.conc,
        area_grid=NORTH_AREA_GRID if hemisphere == 'north' else SOUTH_AREA_GRID,
    )

    fig, ax = plt.subplots(
        nrows=2, ncols=1, subplot_kw={'aspect': 'auto', 'autoscale_on': True}
    )

    _ax = ax[0]

    _ax.plot(
        amsr2_cdr_extent.date,
        amsr2_cdr_extent.data,
        label=f'AMSR2 (AU_SI{resolution}) CDR',
    )
    _ax.plot(cdr_extent.date, cdr_extent.data, label='CDR')
    max_extent = np.max([cdr_extent.max(), amsr2_cdr_extent.max()])
    _ax.set(
        xlabel='date',
        ylabel='Extent (Millions of square kilometers)',
        title='Extent',
        xlim=(cdr_extent.date.min(), cdr_extent.date.max()),
        yticks=np.arange(0, float(max_extent) + 2, 2),
    )
    _ax.legend()
    _ax.grid()

    _ax = ax[1]
    diff = amsr2_cdr_extent - cdr_extent
    _ax.plot(diff.date, diff.data)
    _ax.set(
        xlabel='date',
        ylabel='Extent (Millions of square kilometers)',
        title='AMSR2 CDR minus CDR extent',
        xlim=(diff.date.min(), diff.date.max()),
    )
    _ax.grid()

    fig.set_size_inches(w=25, h=16)
    fig.savefig(
        OUTPUT_DIR / f'{start_date:%Y%m%d}_{end_date:%Y%m%d}_extent_comparison.png',
        bbox_inches='tight',
        pad_inches=0.05,
    )
