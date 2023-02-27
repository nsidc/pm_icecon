"""Params for AMSR2.

All parameters pulled from `ret_parameters_amsru2.f`.
"""
from pm_icecon.bt._types import Line
from pm_icecon.config.models.bt import (
    TbSetParams,
    WeatherFilterParams,
    WeatherFilterParamsForSeason,
)

AMSR2_NORTH_PARAMS = dict(
    vh37_params=TbSetParams(
        water_tie_point_set=[207.2, 131.9],
        ice_tie_point_set=[256.3, 241.2],
        lnline=Line(offset=-71.99, slope=1.20),
    ),
    v1937_params=TbSetParams(
        water_tie_point_set=[207.2, 182.4],
        ice_tie_point_set=[256.3, 258.9],
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

AMSR2_SOUTH_PARAMS = dict(
    vh37_params=TbSetParams(
        water_tie_point_set=[207.6, 131.9],
        ice_tie_point_set=[259.4, 247.3],
        lnline=Line(offset=-90.62, slope=1.2759),
    ),
    v1937_params=TbSetParams(
        water_tie_point_set=[207.6, 182.7],
        ice_tie_point_set=[259.4, 261.6],
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
