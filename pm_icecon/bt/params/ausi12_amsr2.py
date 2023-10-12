"""Routines to yield parameters for Bootstrap.

Bootstrap Parameters for AMSR2 were adapted from Goddard parameters for AUSI12
input for use with the CDR, but use weather coefficient paramters derived at
NSIDC.

TODO: use the params from `ausi_amsr2.py` as the source of these paramters, and
then override the values we need to change for the CDR (like what is done in
`ausi_amsr2.py` inheriting from the goddard CLASS params.

This code is _only_ used in the ecdr.
"""

import datetime as dt

from pm_icecon.bt.compute_bt_ic import _get_wx_params as interpolate_bt_wx_params
from pm_icecon.bt.params.ausi_amsr2 import (
    GODDARD_AMSR2_NORTH_PARAMS,
    GODDARD_AMSR2_SOUTH_PARAMS,
)
from pm_icecon.config.models.bt import WeatherFilterParams, WeatherFilterParamsForSeason
from pm_icecon.gridid import get_gridid_hemisphere

BOOTSTRAP_PARAMS_INITIAL_AMSR2_NORTH = dict(
    bt_wtp_v37=GODDARD_AMSR2_NORTH_PARAMS['vh37_params'].water_tie_point_set[0],
    bt_wtp_h37=GODDARD_AMSR2_NORTH_PARAMS['vh37_params'].water_tie_point_set[1],
    bt_wtp_v19=GODDARD_AMSR2_NORTH_PARAMS['v1937_params'].water_tie_point_set[1],
    bt_itp_v37=GODDARD_AMSR2_NORTH_PARAMS['vh37_params'].ice_tie_point_set[0],
    bt_itp_h37=GODDARD_AMSR2_NORTH_PARAMS['vh37_params'].ice_tie_point_set[1],
    bt_itp_v19=GODDARD_AMSR2_NORTH_PARAMS['v1937_params'].ice_tie_point_set[1],
    vh37_lnline=GODDARD_AMSR2_NORTH_PARAMS['vh37_params'].lnline,
    v1937_lnline=GODDARD_AMSR2_NORTH_PARAMS['v1937_params'].lnline,
    weather_filter_seasons=[
        # November through April (`seas=1` in `boot_ice_amsru2_np.f`)
        WeatherFilterParamsForSeason(
            start_month=11,
            end_month=4,
            weather_filter_params=WeatherFilterParams(
                wintrc=84.73,
                wslope=0.5352,
                wxlimt=13.7,
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
                wxlimt=21.7,
            ),
        ),
        # October (`seas=4`) will get interpolated from the previous and next
        # (first in this list) season.
    ],
)

BOOTSTRAP_PARAMS_INITIAL_AMSR2_SOUTH = dict(
    bt_wtp_v37=GODDARD_AMSR2_SOUTH_PARAMS['vh37_params'].water_tie_point_set[0],
    bt_wtp_h37=GODDARD_AMSR2_SOUTH_PARAMS['vh37_params'].water_tie_point_set[1],
    bt_wtp_v19=GODDARD_AMSR2_SOUTH_PARAMS['v1937_params'].water_tie_point_set[1],
    bt_itp_v37=GODDARD_AMSR2_SOUTH_PARAMS['vh37_params'].ice_tie_point_set[0],
    bt_itp_h37=GODDARD_AMSR2_SOUTH_PARAMS['vh37_params'].ice_tie_point_set[1],
    bt_itp_v19=GODDARD_AMSR2_SOUTH_PARAMS['v1937_params'].ice_tie_point_set[1],
    vh37_lnline=GODDARD_AMSR2_SOUTH_PARAMS['vh37_params'].lnline,
    v1937_lnline=GODDARD_AMSR2_SOUTH_PARAMS['v1937_params'].lnline,
    weather_filter_seasons=[
        # Just one season for the S. hemisphere.
        WeatherFilterParamsForSeason(
            start_month=1,
            end_month=12,
            weather_filter_params=WeatherFilterParams(
                wintrc=85.13,
                wslope=0.5379,
                wxlimt=14.3,
            ),
        ),
    ],  # noqa (ignore "not used" flake8 warning)
)


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
