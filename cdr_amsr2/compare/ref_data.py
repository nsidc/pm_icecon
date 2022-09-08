"""Functions for fetching and interacting with reference data.

For comparison and validation purposes, it is useful to compare the outputs from
our code against other sea ice concentration products.
"""
import datetime as dt

import xarray as xr
from pyresample import AreaDefinition
from pyresample.image import ImageContainerNearest
from seaice.data.api import concentration_daily
from seaice.nasateam import NORTH, SOUTH

from cdr_amsr2._types import Hemisphere
from cdr_amsr2.util import get_ps12_grid_shape, get_ps25_grid_shape


def _get_area_def(*, hemisphere: Hemisphere, shape: tuple[int, int]) -> AreaDefinition:
    proj_id = {
        'north': 'EPSG:3411',
        'south': 'EPSG:3412',
    }[hemisphere]

    proj_str = {
        'north': (
            '+proj=stere +lat_0=90 +lat_ts=70 +lon_0=-45 +k=1'
            ' +x_0=0 +y_0=0 +a=6378273 +b=6356889.449 +units=m +no_defs'
        ),
        'south': (
            '+proj=stere +lat_0=-90 +lat_ts=-70 +lon_0=0 +k=1'
            ' +x_0=0 +y_0=0 +a=6378273 +b=6356889.449 +units=m +no_defs'
        ),
    }[hemisphere]

    # (lower_left_x, lower_left_y, upper_right_x, upper_right_y)
    area_extent = {
        'north': (-3850000.0, -5350000.0, 3750000.0, 5850000.0),
        'south': (-3950000.0, -3950000.0, 3950000.0, 4350000.0),
    }[hemisphere]

    area_def = AreaDefinition(
        area_id=hemisphere,
        description='Polarstereo North 25km',
        proj_id=proj_id,
        projection=proj_str,
        width=shape[1],
        height=shape[0],
        area_extent=area_extent,
    )

    return area_def


def get_sea_ice_index(
    *, hemisphere: Hemisphere, date: dt.date, resolution='25'
) -> xr.Dataset:
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

    data = gridset['data']

    if resolution not in ('12', '25'):
        raise NotImplementedError()
    if resolution == '12':
        src_area = _get_area_def(
            hemisphere=hemisphere, shape=get_ps25_grid_shape(hemisphere=hemisphere)
        )
        dst_area = _get_area_def(
            hemisphere=hemisphere,
            shape=get_ps12_grid_shape(hemisphere=hemisphere),
        )
        # TODO: this will bilinearly interpolate flag values as well. Need to
        # mask those out. For now, Use NN resampling.
        # data = ImageContainerBilinear(data, src_area).resample(dst_area).image_data
        data = (
            ImageContainerNearest(
                data,
                src_area,
                radius_of_influence=25000,
            )
            .resample(dst_area)
            .image_data
        )

    conc_ds = xr.Dataset({'conc': (('y', 'x'), data)})

    # 'flip' the data. NOTE/TODO: this should not be necessary. Can probably
    # pass the correct coords to `xr.Dataset` above and in the other places we
    # create xr datasets.
    conc_ds = conc_ds.reindex(y=conc_ds.y[::-1], x=conc_ds.x)

    return conc_ds
