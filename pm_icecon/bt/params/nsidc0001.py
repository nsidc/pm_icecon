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

"""
NSIDC0001_F17_NORTH_PARAMS = ParamsDict(
    vh37_params=TbSetParams(
        water_tie_point_set=(Tiepoint(201.916), Tiepoint(132.815)),
        ice_tie_point_set=(Tiepoint(255.670), Tiepoint(241.713)),
        lnline=Line(offset=-73.5471, slope=1.21104),
        # iceline=Line(offset=-25.9729, slope=1.04382),  # not yet used?
    ),
    v1937_params=TbSetParams(
        water_tie_point_set=(Tiepoint(201.916), Tiepoint(178.771)),
        ice_tie_point_set=(Tiepoint(255.670), Tiepoint(258.341)),
        lnline=Line(offset=47.0061, slope=0.809335),
        # iceline=Line(offset=112.803, slope=0.550296),  # not yet used?
    ),
    # NSIDC-0001 SSMI and SSMIS have transition months May and Oct
    weather_filter_seasons=[
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
    ],
)

NSIDC0001_F17_SOUTH_PARAMS = ParamsDict(
    vh37_params=TbSetParams(
        water_tie_point_set=(Tiepoint(201.990), Tiepoint(133.943)),
        ice_tie_point_set=(Tiepoint(259.122), Tiepoint(248.284)),
        lnline=Line(offset=-90.9384, slope=1.28239),
        # iceline=Line(offset=-40.8250, slope=1.11404),  # not yet used?
    ),
    v1937_params=TbSetParams(
        water_tie_point_set=(Tiepoint(201.990), Tiepoint(178.358)),
        ice_tie_point_set=(Tiepoint(259.122), Tiepoint(261.654)),
        lnline=Line(offset=61.7438, slope=0.767205),
        # iceline=Line(offset=114.825, slope=0.570622),  # not yet used?
    ),
    weather_filter_seasons=[
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
"""

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

# TODO: icelines do not appear to be implemented in BT code yet
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
    # vh37_iceline=Line(offset=-25.9729, slope=1.04382),
    # v1937_iceline=Line(offset=112.803, slope=0.550296),
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
    # vh37_iceline=Line(offset=-40.8250, slope=1.11404),
    # v1937_iceline=Line(offset=114.825, slope=0.570622),
    # weather_filter_seasons should be added by sensor
)

"""
# REPLACED BY sensor-dependent code
# TODO: Is this where we would overwrite the default values?
BOOTSTRAP_PARAMS_INITIAL_F17_NORTH = dict(
    bt_wtp_v37=NSIDC0001_F17_NORTH_PARAMS["vh37_params"].water_tie_point_set[0],
    bt_wtp_h37=NSIDC0001_F17_NORTH_PARAMS["vh37_params"].water_tie_point_set[1],
    bt_wtp_v19=NSIDC0001_F17_NORTH_PARAMS["v1937_params"].water_tie_point_set[1],
    bt_itp_v37=NSIDC0001_F17_NORTH_PARAMS["vh37_params"].ice_tie_point_set[0],
    bt_itp_h37=NSIDC0001_F17_NORTH_PARAMS["vh37_params"].ice_tie_point_set[1],
    bt_itp_v19=NSIDC0001_F17_NORTH_PARAMS["v1937_params"].ice_tie_point_set[1],
    vh37_lnline=NSIDC0001_F17_NORTH_PARAMS["vh37_params"].lnline,
    v1937_lnline=NSIDC0001_F17_NORTH_PARAMS["v1937_params"].lnline,
    weather_filter_seasons=NSIDC0001_F17_NORTH_PARAMS["weather_filter_seasons"],
)

BOOTSTRAP_PARAMS_INITIAL_F17_SOUTH = dict(
    bt_wtp_v37=NSIDC0001_F17_SOUTH_PARAMS["vh37_params"].water_tie_point_set[0],
    bt_wtp_h37=NSIDC0001_F17_SOUTH_PARAMS["vh37_params"].water_tie_point_set[1],
    bt_wtp_v19=NSIDC0001_F17_SOUTH_PARAMS["v1937_params"].water_tie_point_set[1],
    bt_itp_v37=NSIDC0001_F17_SOUTH_PARAMS["vh37_params"].ice_tie_point_set[0],
    bt_itp_h37=NSIDC0001_F17_SOUTH_PARAMS["vh37_params"].ice_tie_point_set[1],
    bt_itp_v19=NSIDC0001_F17_SOUTH_PARAMS["v1937_params"].ice_tie_point_set[1],
    vh37_lnline=NSIDC0001_F17_SOUTH_PARAMS["vh37_params"].lnline,
    v1937_lnline=NSIDC0001_F17_SOUTH_PARAMS["v1937_params"].lnline,
    weather_filter_seasons=NSIDC0001_F17_SOUTH_PARAMS["weather_filter_seasons"],
)
"""


# Rename this...without F17!
def get_F17_bootstrap_params(
    *,
    date: dt.date,
    satellite: str,
    gridid: str,
) -> dict:
    """Assign the bootstrap parameters for this date, sat, grid."""
    hemisphere = get_gridid_hemisphere(gridid)
    # TODO: Other satellites -- eg F13, F8 --  will need to be added here
    SMMR_SAT_LIST = ("n07",)
    SSMI_SAT_LIST = (
        "F08",
        "F11",
        "F13",
    )
    SSMIS_SAT_LIST = ("F17",)
    if satellite in SMMR_SAT_LIST:
        if hemisphere == "north":
            initial_bt_params = NSIDC0001_BASE_PARAMS_NORTH.copy()
            initial_bt_params[
                "weather_filter_seasons"
            ] = WEATHER_FILTER_SEASONS_SMMR_NORTH
        elif hemisphere == "south":
            initial_bt_params = NSIDC0001_BASE_PARAMS_SOUTH.copy()
            initial_bt_params[
                "weather_filter_seasons"
            ] = WEATHER_FILTER_SEASONS_SMMR_SOUTH
    elif satellite in SSMI_SAT_LIST:
        if hemisphere == "north":
            initial_bt_params = NSIDC0001_BASE_PARAMS_NORTH.copy()
            initial_bt_params[
                "weather_filter_seasons"
            ] = WEATHER_FILTER_SEASONS_SSMI_NORTH
        elif hemisphere == "south":
            initial_bt_params = NSIDC0001_BASE_PARAMS_SOUTH.copy()
            initial_bt_params[
                "weather_filter_seasons"
            ] = WEATHER_FILTER_SEASONS_SSMI_SOUTH
    elif satellite in SSMIS_SAT_LIST:
        if hemisphere == "north":
            initial_bt_params = NSIDC0001_BASE_PARAMS_NORTH.copy()
            initial_bt_params[
                "weather_filter_seasons"
            ] = WEATHER_FILTER_SEASONS_SSMIS_NORTH
        elif hemisphere == "south":
            initial_bt_params = NSIDC0001_BASE_PARAMS_SOUTH.copy()
            initial_bt_params[
                "weather_filter_seasons"
            ] = WEATHER_FILTER_SEASONS_SSMIS_SOUTH
    else:
        raise ValueError(
            f"Bootstrap params not yet defined for:\n  satellite: {satellite}"
        )

    bt_params = setup_bootstrap_params_dict(
        initial_params_dict=initial_bt_params, date=date
    )

    return bt_params
