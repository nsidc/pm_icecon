"""Routines to yield parameters for Bootstrap.

Bootstrap Parameters for AMSR2 were taken from `ret_parameters_amsru2.f`

WIP parameters for AU_SI12 AMSR2 data used by the CDR.

This differs from what's used in the ECDR, which is defined in
`bt.params.ausi12_amsr2.py`
"""

import datetime as dt

import numpy as np

from pm_icecon.bt._types import Line
from pm_icecon.bt.compute_bt_ic import _get_wx_params as interpolate_bt_wx_params
from pm_icecon.bt.masks import get_ps_invalid_ice_mask
from pm_icecon.config.models.bt import (
    BootstrapParams,
    TbSetParams,
    WeatherFilterParams,
    WeatherFilterParamsForSeason,
    cast_as_TiepointSet,
)
from pm_icecon.masks import get_ps_land_mask, get_ps_pole_hole_mask

BOOTSTRAP_PARAMS_INITIAL_AMSR2_NORTH = dict(
    bt_wtp_v37=207.2,
    bt_wtp_h37=131.9,
    bt_wtp_v19=182.4,
    bt_itp_v37=256.3,
    bt_itp_h37=241.2,
    bt_itp_v19=258.9,
    vh37_lnline=Line(offset=-71.99, slope=1.20),
    v1937_lnline=Line(offset=48.26, slope=0.8048),
    weather_filter_seasons=[
        # November through April (`seas=1` in `boot_ice_amsru2_np.f`)
        WeatherFilterParamsForSeason(
            start_month=11,
            end_month=4,
            weather_filter_params=WeatherFilterParams(
                wintrc=84.73,
                wslope=0.5352,
                wxlimt=18.39,
            ),
        ),
        # May (`seas=2`) will get interpolated from the previous and next season
        # June through Sept. (`seas=3`)
        WeatherFilterParamsForSeason(
            start_month=6,
            end_month=9,
            weather_filter_params=WeatherFilterParams(
                wintrc=82.71,
                wslope=0.5352,
                wxlimt=23.34,
            ),
        ),
        # October (`seas=4`) will get interpolated from the previous and next
        # (first in this list) season.
    ],
)

BOOTSTRAP_PARAMS_INITIAL_AMSR2_SOUTH = dict(
    bt_wtp_v37=207.6,
    bt_wtp_h37=131.9,
    bt_wtp_v19=182.7,
    bt_itp_v37=259.4,
    bt_itp_h37=247.3,
    bt_itp_v19=261.6,
    vh37_lnline=Line(offset=-90.62, slope=1.2759),
    v1937_lnline=Line(offset=62.89, slope=0.7618),
    weather_filter_seasons=[
        # Just one season for the S. hemisphere.
        WeatherFilterParamsForSeason(
            start_month=1,
            end_month=12,
            weather_filter_params=WeatherFilterParams(
                wintrc=85.13,
                wslope=0.5379,
                wxlimt=18.596,
            ),
        ),
    ],  # noqa (ignore "not used" flake8 warning)
)


# Note: these routines are also in seaice_ecdr's gridid_to_xr_dataarray.py
def get_gridid_hemisphere(gridid):
    # Return the hemisphere of the gridid
    if 'psn' in gridid:
        return 'north'
    elif 'e2n' in gridid:
        return 'north'
    elif 'pss' in gridid:
        return 'south'
    elif 'e2s' in gridid:
        return 'south'
    else:
        raise ValueError(f'Could not find hemisphere for gridid: {gridid}')


def get_gridid_resolution(gridid):
    # Return the hemisphere of the gridid
    if '3.125' in gridid:
        return '3.125'
    elif '6.25' in gridid:
        return '6.25'
    elif '12.5' in gridid:
        return '12.5'
    elif '25' in gridid:
        return '25'
    else:
        raise ValueError(f'Could not find resolution for gridid: {gridid}')


def convert_to_pmicecon_bt_params(hemisphere, params, fields) -> BootstrapParams:
    """Convert to old-style bt_params."""
    oldstyle_bt_params = BootstrapParams(
        land_mask=np.array(fields['land_mask']).squeeze(),
        # There's no pole hole in the southern hemisphere.
        pole_mask=np.array(fields['pole_mask']) if hemisphere == 'north' else None,
        invalid_ice_mask=np.array(fields['invalid_ice_mask']),
        vh37_params=TbSetParams(
            water_tie_point_set=cast_as_TiepointSet(
                params['bt_wtp_v37'], params['bt_wtp_h37']
            ),
            ice_tie_point_set=cast_as_TiepointSet(
                params['bt_itp_v37'], params['bt_itp_h37']
            ),
            lnline=params['vh37_lnline'],
        ),
        v1937_params=TbSetParams(
            water_tie_point_set=cast_as_TiepointSet(
                params['bt_wtp_v37'], params['bt_wtp_v19']
            ),
            ice_tie_point_set=cast_as_TiepointSet(
                params['bt_itp_v37'], params['bt_itp_v19']
            ),
            lnline=params['v1937_lnline'],
        ),
        **params,
    )
    return oldstyle_bt_params


def get_bootstrap_params(
    *,
    date: dt.date,
    satellite: str,
    gridid: str,
) -> dict:
    """Assign the bootstrap parameters for this date, sat, grid."""
    hemisphere = get_gridid_hemisphere(gridid)
    if satellite == 'amsr2':
        if hemisphere == 'north':
            bt_params = BOOTSTRAP_PARAMS_INITIAL_AMSR2_NORTH
        elif hemisphere == 'south':
            bt_params = BOOTSTRAP_PARAMS_INITIAL_AMSR2_SOUTH
        else:
            raise ValueError(
                'Could not initialize Bootstrap params for:\n'
                f'satellite: {satellite}\n  hemisphere: {hemisphere}'
            )
    else:
        raise ValueError(
            f'Bootstrap params not yet definted for:\n  satellite: {satellite}'
        )

    # Set standard bootstrap values
    bt_params['add1'] = 0.0
    bt_params['add2'] = -2.0
    bt_params['minic'] = 10.0
    bt_params['maxic'] = 1.0
    bt_params['mintb'] = 10.0
    bt_params['maxtb'] = 320.0

    # Some definitions include seasonal values for wintrc, wslope, wxlimt
    if 'wintrc' not in bt_params.keys():
        # weather_filter_seasons = bt_params['weather_filter_seasons']
        wfs = bt_params['weather_filter_seasons']
        bt_weather_params_struct = interpolate_bt_wx_params(
            date=date,
            weather_filter_seasons=wfs,  # type: ignore
        )
        bt_params['wintrc'] = bt_weather_params_struct.wintrc
        bt_params['wslope'] = bt_weather_params_struct.wslope
        bt_params['wxlimt'] = bt_weather_params_struct.wxlimt

    return bt_params


def get_bootstrap_fields(
    *,
    date: dt.date,
    satellite: str,
    gridid: str,
):
    hemisphere = get_gridid_hemisphere(gridid)
    resolution = get_gridid_resolution(gridid)
    # Modification for inexact resolution
    if resolution == '12.5':
        resolution = '12'

    invalid_ice_mask = get_ps_invalid_ice_mask(
        hemisphere=hemisphere,
        date=date,
        resolution=resolution,  # type: ignore[arg-type]
    )

    land_mask = get_ps_land_mask(hemisphere=hemisphere, resolution=resolution)

    # There's no pole hole in the southern hemisphere.
    pole_mask = (
        get_ps_pole_hole_mask(resolution=resolution) if hemisphere == 'north' else None
    )

    return dict(
        invalid_ice_mask=invalid_ice_mask,
        land_mask=land_mask,
        pole_mask=pole_mask,
    )
