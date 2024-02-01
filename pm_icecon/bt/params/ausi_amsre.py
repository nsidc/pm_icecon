"""AMSRE Bootstrap parameters.

Bootstrap parameters for use with AMSRE derived from AU_SI products

For now, these parameters are copied from the AMSR2 Bootstrap parameters
CDR.
"""
import datetime as dt
from typing import cast

from pm_tb_data._types import Hemisphere

from pm_icecon.bt._types import Line, Tiepoint
from pm_icecon.bt.params._types import ParamsDict
from pm_icecon.bt.params.util import setup_bootstrap_params_dict
from pm_icecon.config.models.bt import (
    BootstrapParams,
    TbSetParams,
    WeatherFilterParams,
    WeatherFilterParamsForSeason,
)
from pm_icecon.gridid import get_gridid_hemisphere

GODDARD_AMSRE_NORTH_PARAMS = ParamsDict(
    vh37_params=TbSetParams(
        water_tie_point_set=(Tiepoint(207.2), Tiepoint(131.9)),
        ice_tie_point_set=(Tiepoint(256.3), Tiepoint(241.2)),
        lnline=Line(offset=-71.99, slope=1.20),
    ),
    v1937_params=TbSetParams(
        water_tie_point_set=(Tiepoint(207.2), Tiepoint(182.4)),
        ice_tie_point_set=(Tiepoint(256.3), Tiepoint(258.9)),
        lnline=Line(offset=48.26, slope=0.8048),
    ),
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

GODDARD_AMSRE_SOUTH_PARAMS = ParamsDict(
    vh37_params=TbSetParams(
        water_tie_point_set=(Tiepoint(207.6), Tiepoint(131.9)),
        ice_tie_point_set=(Tiepoint(259.4), Tiepoint(247.3)),
        lnline=Line(offset=-90.62, slope=1.2759),
    ),
    v1937_params=TbSetParams(
        water_tie_point_set=(Tiepoint(207.6), Tiepoint(182.7)),
        ice_tie_point_set=(Tiepoint(259.4), Tiepoint(261.6)),
        lnline=Line(offset=62.89, slope=0.7618),
    ),
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
    ],
)

BOOTSTRAP_PARAMS_INITIAL_AMSRE_NORTH = dict(
    bt_wtp_v37=GODDARD_AMSRE_NORTH_PARAMS["vh37_params"].water_tie_point_set[0],
    bt_wtp_h37=GODDARD_AMSRE_NORTH_PARAMS["vh37_params"].water_tie_point_set[1],
    bt_wtp_v19=GODDARD_AMSRE_NORTH_PARAMS["v1937_params"].water_tie_point_set[1],
    bt_itp_v37=GODDARD_AMSRE_NORTH_PARAMS["vh37_params"].ice_tie_point_set[0],
    bt_itp_h37=GODDARD_AMSRE_NORTH_PARAMS["vh37_params"].ice_tie_point_set[1],
    bt_itp_v19=GODDARD_AMSRE_NORTH_PARAMS["v1937_params"].ice_tie_point_set[1],
    vh37_lnline=GODDARD_AMSRE_NORTH_PARAMS["vh37_params"].lnline,
    v1937_lnline=GODDARD_AMSRE_NORTH_PARAMS["v1937_params"].lnline,
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

BOOTSTRAP_PARAMS_INITIAL_AMSRE_SOUTH = dict(
    bt_wtp_v37=GODDARD_AMSRE_SOUTH_PARAMS["vh37_params"].water_tie_point_set[0],
    bt_wtp_h37=GODDARD_AMSRE_SOUTH_PARAMS["vh37_params"].water_tie_point_set[1],
    bt_wtp_v19=GODDARD_AMSRE_SOUTH_PARAMS["v1937_params"].water_tie_point_set[1],
    bt_itp_v37=GODDARD_AMSRE_SOUTH_PARAMS["vh37_params"].ice_tie_point_set[0],
    bt_itp_h37=GODDARD_AMSRE_SOUTH_PARAMS["vh37_params"].ice_tie_point_set[1],
    bt_itp_v19=GODDARD_AMSRE_SOUTH_PARAMS["v1937_params"].ice_tie_point_set[1],
    vh37_lnline=GODDARD_AMSRE_SOUTH_PARAMS["vh37_params"].lnline,
    v1937_lnline=GODDARD_AMSRE_SOUTH_PARAMS["v1937_params"].lnline,
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


CDR_AMSRE_NORTH_PARAMS = GODDARD_AMSRE_NORTH_PARAMS.copy()
_bt_north_weather_filter_seasons = BOOTSTRAP_PARAMS_INITIAL_AMSRE_NORTH[
    "weather_filter_seasons"
]
_bt_north_weather_filter_seasons = cast(
    list[WeatherFilterParamsForSeason], _bt_north_weather_filter_seasons
)
CDR_AMSRE_NORTH_PARAMS["weather_filter_seasons"] = _bt_north_weather_filter_seasons

CDR_AMSRE_SOUTH_PARAMS = GODDARD_AMSRE_SOUTH_PARAMS.copy()
_bt_south_weather_filter_seasons = BOOTSTRAP_PARAMS_INITIAL_AMSRE_SOUTH[
    "weather_filter_seasons"
]
_bt_south_weather_filter_seasons = cast(
    list[WeatherFilterParamsForSeason], _bt_south_weather_filter_seasons
)
CDR_AMSRE_SOUTH_PARAMS["weather_filter_seasons"] = _bt_south_weather_filter_seasons


def get_amsre_params(
    *,
    hemisphere: Hemisphere,
) -> BootstrapParams:
    bt_params = BootstrapParams(
        **(CDR_AMSRE_NORTH_PARAMS if hemisphere == "north" else CDR_AMSRE_SOUTH_PARAMS),
    )

    return bt_params


def get_ausi_amsre_bootstrap_params(
    *,
    date: dt.date,
    satellite: str,
    gridid: str,
) -> dict:
    """Assign the bootstrap parameters for this date, sat, grid."""
    hemisphere = get_gridid_hemisphere(gridid)
    if satellite == "amsre":
        if hemisphere == "north":
            initial_bt_params = BOOTSTRAP_PARAMS_INITIAL_AMSRE_NORTH
        elif hemisphere == "south":
            initial_bt_params = BOOTSTRAP_PARAMS_INITIAL_AMSRE_SOUTH
        else:
            raise ValueError(
                "Could not initialize Bootstrap params for:\n"
                f"satellite: {satellite}\n  hemisphere: {hemisphere}"
            )
    else:
        raise ValueError(
            f"Bootstrap params not yet defined for:\n  satellite: {satellite}"
        )

    bt_params = setup_bootstrap_params_dict(
        initial_params_dict=initial_bt_params, date=date
    )

    return bt_params
