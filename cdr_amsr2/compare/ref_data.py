"""Functions for fetching and interacting with reference data.

For comparison and validation purposes, it is useful to compare the outputs from
our code against other sea ice concentration products.
"""
import datetime as dt

import xarray as xr
from seaice.data.api import concentration_daily
from seaice.nasateam import NORTH, SOUTH

from cdr_amsr2._types import Hemisphere


def get_sea_ice_index(*, hemisphere: Hemisphere, date: dt.date) -> xr.Dataset:
    """Return a sea ice concentration field from 0051 or 0081.

    Requires the environment variables `EARTHDATA_USERNAME` and
    `EARTHDATA_PASSWORD` to be set. Assumes access to ECS datapools on NSIDC's
    virtual machine infrastructure (e.g., `/ecs/DP1/`).

    Concentrations are floating point values 0-100
    """
    gridset = concentration_daily(
        hemisphere=NORTH if hemisphere == 'north' else SOUTH,
        year=date.year,
        month=date.month,
        day=date.day,
        allow_empty_gridset=False,
    )

    conc_ds = xr.Dataset({'conc': (('y', 'x'), gridset['data'])})

    # 'flip' the data. NOTE/TODO: this should not be necessary. Can probably
    # pass the correct coords to `xr.Dataset` above and in the other places we
    # create xr datasets.
    conc_ds = conc_ds.reindex(y=conc_ds.y[::-1], x=conc_ds.x)

    return conc_ds
