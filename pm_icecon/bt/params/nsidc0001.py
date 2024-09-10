"""NSIDC0001 Bootstrap parameters.

Bootstrap parameters for use with TBs derived from NSIDC-0001

Parameters are based on values from Bootstrap code in cdralgos:
  https://bitbucket.org/nsidc/cdralgos/src/master/bt_cdr/ret_parameters_cdr.f
"""

import datetime as dt

from pm_icecon.bt._types import Line
from pm_icecon.bt.params.util import setup_bootstrap_params_dict
from pm_icecon.config.models.bt import (
    WeatherFilterParams,
    WeatherFilterParamsForSeason,
)
from pm_icecon.gridid import get_gridid_hemisphere

# weather filter for SSMIS, eg F17
WEATHER_FILTER_SEASONS_SSMIS_NORTH = [
    # weather_filter_seasons=
    # November through April (`seas=1` in `ret_parameters_cdr.f`)
    WeatherFilterParamsForSeason(
        start_month=11,
        end_month=4,
        weather_filter_params=WeatherFilterParams(
            wintrc=87.6467,
            wslope=0.517333,
            wxlimt=14.00,
        ),
    ),
    # May (`seas=2`) will get interpolated from the previous and next season
    # June through Sept. (`seas=3`)
    WeatherFilterParamsForSeason(
        start_month=6,
        end_month=9,
        weather_filter_params=WeatherFilterParams(
            wintrc=89.20000,
            wslope=0.503750,
            wxlimt=21.00,
        ),
    ),
    # October (`seas=4`) will get interpolated from the previous and next
    # (first in this list) season.
]

WEATHER_FILTER_SEASONS_SSMIS_SOUTH = [
    # weather_filter_seasons=[
    # Just one season for the S. hemisphere.
    WeatherFilterParamsForSeason(
        start_month=1,
        end_month=12,
        weather_filter_params=WeatherFilterParams(
            wintrc=93.2861,
            wslope=0.497374,
            wxlimt=16.5,
        ),
    ),
]

WEATHER_FILTER_SEASONS_SSMIS_SOUTH_DICT = dict(
    wintrc=93.2861,
    wslope=0.497374,
    wxlimt=16.5,
)

# weather filter for SSMI, eg F08, F11, F13
WEATHER_FILTER_SEASONS_SSMI_NORTH = [
    # November through April (`seas=1` in `ret_parameters_cdr.f`)
    WeatherFilterParamsForSeason(
        start_month=11,
        end_month=4,
        weather_filter_params=WeatherFilterParams(
            wintrc=90.3355,
            wslope=0.501537,
            wxlimt=14.00,
        ),
    ),
    # May (`seas=2`) will get interpolated from the previous and next season
    # June through Sept. (`seas=3`)
    WeatherFilterParamsForSeason(
        start_month=6,
        end_month=9,
        weather_filter_params=WeatherFilterParams(
            wintrc=89.3316,
            wslope=0.501537,
            wxlimt=21.00,
        ),
    ),
    # October (`seas=4`) will get interpolated from the previous and next
    # (first in this list) season.
]

WEATHER_FILTER_SEASONS_SSMI_SOUTH = (
    [
        # Note: these are the same for SSMI and SSMIS
        # weather_filter_seasons=[
        # Just one season for the S. hemisphere.
        WeatherFilterParamsForSeason(
            start_month=1,
            end_month=12,
            weather_filter_params=WeatherFilterParams(
                wintrc=93.2861,
                wslope=0.497374,
                wxlimt=16.5,
            ),
        ),
    ],
)
WEATHER_FILTER_SEASONS_SSMI_SOUTH_DICT = dict(
    wintrc=93.2861,
    wslope=0.497374,
    wxlimt=16.5,
)

# TODO: icelines appear to be used in BT code after being calculated,
#       so these initial values are ignored?  They are included here
#       to match cdralgos code
# NOTE: weather_filter_seasons value should be added by sensor
NSIDC0001_BASE_PARAMS_NORTH = dict(
    bt_wtp_v37=201.916,
    bt_wtp_h37=132.815,
    bt_wtp_v19=178.771,
    bt_itp_v37=255.670,
    bt_itp_h37=241.713,
    bt_itp_v19=258.341,
    vh37_lnline=Line(offset=-73.5471, slope=1.21104),
    v1937_lnline=Line(offset=47.0061, slope=0.809335),
    vh37_iceline=Line(offset=-25.9729, slope=1.04382),
    v1937_iceline=Line(offset=112.803, slope=0.550296),
    # weather_filter_seasons should be added by sensor,
)

NSIDC0001_BASE_PARAMS_SOUTH = dict(
    bt_wtp_v37=201.990,
    bt_wtp_h37=133.943,
    bt_wtp_v19=178.358,
    bt_itp_v37=259.122,
    bt_itp_h37=248.284,
    bt_itp_v19=261.654,
    vh37_lnline=Line(offset=-90.9384, slope=1.28239),
    v1937_lnline=Line(offset=61.7438, slope=0.767205),
    vh37_iceline=Line(offset=-40.8250, slope=1.11404),
    v1937_iceline=Line(offset=114.825, slope=0.570622),
    # weather_filter_seasons should be added by sensor
)


def get_nsidc0001_bootstrap_params(
    *,
    date: dt.date,
    satellite: str,
    gridid: str,
) -> dict:
    """Assign the bootstrap parameters for this date, sat, grid."""
    hemisphere = get_gridid_hemisphere(gridid)
    SSMI_SAT_LIST = (
        "F08",
        "F11",
        "F13",
    )
    SSMIS_SAT_LIST = ("F17",)
    if satellite in SSMI_SAT_LIST:
        if hemisphere == "north":
            initial_bt_params = NSIDC0001_BASE_PARAMS_NORTH.copy()
            initial_bt_params["weather_filter_seasons"] = (
                WEATHER_FILTER_SEASONS_SSMI_NORTH
            )
        elif hemisphere == "south":
            initial_bt_params = NSIDC0001_BASE_PARAMS_SOUTH.copy()
            initial_bt_params = (
                initial_bt_params | WEATHER_FILTER_SEASONS_SSMI_SOUTH_DICT
            )
            # initial_bt_params[
            #     "weather_filter_seasons"
            # ] = WEATHER_FILTER_SEASONS_SSMI_SOUTH
    elif satellite in SSMIS_SAT_LIST:
        if hemisphere == "north":
            initial_bt_params = NSIDC0001_BASE_PARAMS_NORTH.copy()
            initial_bt_params["weather_filter_seasons"] = (
                WEATHER_FILTER_SEASONS_SSMIS_NORTH
            )
        elif hemisphere == "south":
            initial_bt_params = NSIDC0001_BASE_PARAMS_SOUTH.copy()
            initial_bt_params = (
                initial_bt_params | WEATHER_FILTER_SEASONS_SSMIS_SOUTH_DICT
            )
            # initial_bt_params[
            #     "weather_filter_seasons"
            # ] = WEATHER_FILTER_SEASONS_SSMIS_SOUTH
    else:
        raise ValueError(
            f"Bootstrap params not yet defined for:\n  satellite: {satellite}"
        )

    bt_params = setup_bootstrap_params_dict(
        initial_params_dict=initial_bt_params, date=date
    )

    # Per cdralgos routine calc_bt_params.f, add1 and add2 differ for SH
    if hemisphere == "south":
        bt_params["add1"] = 4.0
        bt_params["add2"] = 0.0

    return bt_params
