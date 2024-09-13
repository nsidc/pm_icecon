"""NSIDC0007 Bootstrap parameters.

TODO: bootstrap parameters are being refactored. See the `nsidc0001` bt params
module for the most up-to-date method/approach.
"""

import datetime as dt

from pm_tb_data._types import Hemisphere

from pm_icecon.bt.params.nsidc0001 import (
    NSIDC0001_BASE_PARAMS_NORTH,
    NSIDC0001_BASE_PARAMS_SOUTH,
)
from pm_icecon.bt.params.util import setup_bootstrap_params_dict
from pm_icecon.config.models.bt import (
    WeatherFilterParams,
    WeatherFilterParamsForSeason,
)

# weather filter for SMMR, eg n07
#  season 1: not-season-2
#  season 2: June 1 - Oct 15 and not-that
# with no transition period between the two seasons
#  and having a season 2 that runs from nov - may
WEATHER_FILTER_SEASONS_SMMR_NORTH = [
    # November through May (`seas=1` in `ret_parameters_cdr.f`)
    WeatherFilterParamsForSeason(
        start_month=10,
        end_month=5,
        start_day=16,
        weather_filter_params=WeatherFilterParams(
            wintrc=53.4153,
            wslope=0.661017,
            wxlimt=22.00,
        ),
    ),
    # June through Sept. (`seas=2`)
    # Note: should run through Oct 15
    WeatherFilterParamsForSeason(
        start_month=6,
        end_month=10,
        end_day=15,
        weather_filter_params=WeatherFilterParams(
            wintrc=60.1667,
            wslope=0.63333,
            wxlimt=24.00,
        ),
    ),
]

WEATHER_FILTER_SEASONS_SMMR_SOUTH = (
    [
        # Just one season for the S. hemisphere.
        WeatherFilterParamsForSeason(
            start_month=1,
            end_month=12,
            weather_filter_params=WeatherFilterParams(
                wintrc=82.500,
                wslope=0.529236,
                wxlimt=24.82,
            ),
        ),
    ],
)

WEATHER_FILTER_SEASONS_SMMR_SOUTH_DICT = dict(
    wintrc=82.500,
    wslope=0.529236,
    wxlimt=24.82,
)


def get_smmr_params(*, hemisphere: Hemisphere, date: dt.date) -> dict:
    if hemisphere == "north":
        initial_bt_params = NSIDC0001_BASE_PARAMS_NORTH.copy()
        initial_bt_params["weather_filter_seasons"] = WEATHER_FILTER_SEASONS_SMMR_NORTH
    else:
        initial_bt_params = NSIDC0001_BASE_PARAMS_SOUTH.copy()
        initial_bt_params = initial_bt_params | WEATHER_FILTER_SEASONS_SMMR_SOUTH_DICT
        # initial_bt_params["weather_filter_seasons"] = WEATHER_FILTER_SEASONS_SMMR_SOUTH

    bt_params = setup_bootstrap_params_dict(
        initial_params_dict=initial_bt_params, date=date
    )
    # Per Fortran code calc_bt_params.f, the add1 and add2 values
    # are different by SMMR or not, and by NH or SH
    # For both SMMR and SSMI, add1=0, add2=-2
    # but both values are different for SH SMMR and SH SSMI
    if hemisphere == "south":
        bt_params["add1"] = 3.75
        bt_params["add2"] = -0.5

    return bt_params
