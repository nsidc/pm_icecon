"""Functions for fetching and interacting with reference data.

For comparison and validation purposes, it is useful to compare the outputs from
our code against other sea ice concentration products.
"""
import datetime as dt
from pathlib import Path

import xarray as xr
from pyresample import AreaDefinition
from pyresample.image import ImageContainerNearest
from seaice.data.api import concentration_daily
from seaice.nasateam import NORTH, SOUTH

from cdr_amsr2._types import Hemisphere
from cdr_amsr2.constants import DEFAULT_FLAG_VALUES
from cdr_amsr2.fetch import au_si
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


def get_au_si_bt_conc(
    *,
    date: dt.date,
    hemisphere: Hemisphere,
    resolution: au_si.AU_SI_RESOLUTIONS,
) -> xr.DataArray:
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

    # change the AU_SI flags to our defaults
    # and make polehole/missing values match ours missing value (110)
    # missing
    bt_conc = bt_conc.where(bt_conc != 110, DEFAULT_FLAG_VALUES.missing)
    # land
    bt_conc = bt_conc.where(bt_conc != 120, DEFAULT_FLAG_VALUES.land)

    return bt_conc


def _find_cdr(
    *,
    date: dt.date,
    hemisphere: Hemisphere,
) -> Path:
    """Find a CDR granule on disk.

    Expects to find 25km resolution CDR granules (netcdf files) in either:

    * /projects/DATASETS/NOAA/G02202_V4
    or
    * /projects/DATASETS/NOAA/G10016_V2

    Prefers finding data from G02202 first.
    """
    base_dir = Path('/projects/DATASETS/NOAA/')
    final_dir = base_dir / 'G02202_V4'
    nrt_dir = base_dir / 'G10016_V2'

    # TODO: icdr has a slightly different fn...
    expected_fn_pattern = f'seaice_conc_daily*_{hemisphere[0]}h_{date:%Y%m%d}_*.nc'
    for data_dir in (final_dir, nrt_dir):
        expected_dir = data_dir / hemisphere / 'daily' / str(date.year)
        results = list(expected_dir.glob(expected_fn_pattern))
        if len(results) == 1:
            return results[0]

        if len(results) > 1:
            raise RuntimeError(
                f'Unexpected number of NC files found for {expected_fn_pattern}'
            )

    raise FileNotFoundError(
        f'No CDR data found for {date=}, {hemisphere=}.'
    )


def get_cdr(
    *,
    date: dt.date,
    hemisphere: Hemisphere,
    resolution: au_si.AU_SI_RESOLUTIONS,
) -> xr.Dataset:
    """Get CDR (G02202 and G10016) concentration field."""
    cdr_fp = _find_cdr(date=date, hemisphere=hemisphere)

    cdr_ds = xr.open_dataset(cdr_fp)
    # Scale the data by 100. Concentrations given as fractions from 0-1.
    cdr_data = cdr_ds['cdr_seaice_conc'].data[0, :, :] * 100

    if resolution == '12':
        src_area = _get_area_def(
            hemisphere=hemisphere, shape=get_ps25_grid_shape(hemisphere=hemisphere)
        )
        dst_area = _get_area_def(
            hemisphere=hemisphere,
            shape=get_ps12_grid_shape(hemisphere=hemisphere),
        )

        cdr_data = (
            ImageContainerNearest(
                cdr_data,
                src_area,
                radius_of_influence=25000,
            )
            .resample(dst_area)
            .image_data
        )


    conc_ds = xr.Dataset({'conc': (('y', 'x'), cdr_data)})

    # 'flip' the data. NOTE/TODO: this should not be necessary. Can probably
    # pass the correct coords to `xr.Dataset` above and in the other places we
    # create xr datasets.
    conc_ds = conc_ds.reindex(y=conc_ds.y[::-1], x=conc_ds.x)

    return conc_ds
