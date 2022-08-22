"""Parameters for DMSP satellites.

Defense Meteorological Satellite Program. E.g, f17, f18.
"""
from cdr_amsr2.config.models.bt import (
    ParaNSB2,
    WeatherFilterParams,
    WeatherFilterParamsForSeason,
)

F17_F18_NORTH_PARAMS = ParaNSB2(
    weather_filter_seasons=[
        # June through October 15
        WeatherFilterParamsForSeason(
            start_month=6,
            end_month=10,
            end_day=15,
            weather_filter_params=WeatherFilterParams(
                wintrc=89.2000,
                wslope=0.503750,
                wxlimt=21.00,
            ),
        ),
        # Oct. 16 through May
        WeatherFilterParamsForSeason(
            start_month=10,
            start_day=16,
            end_month=5,
            weather_filter_params=WeatherFilterParams(
                wintrc=87.6467,
                wslope=0.517333,
                wxlimt=14.00,
            ),
        ),
    ]
)
