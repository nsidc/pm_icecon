"""Params for SMMR (`sat='00'`)."""
from cdr_amsr2.config.models.bt import (
    ParaNSB2,
    WeatherFilterParams,
    WeatherFilterParamsForSeason,
)

SMMR_NORTH_PARAMS = ParaNSB2(
    weather_filter_seasons=[
        # June through October 15
        WeatherFilterParamsForSeason(
            start_month=6,
            end_month=10,
            end_day=15,
            weather_filter_params=WeatherFilterParams(
                wintrc=60.1667,
                wslope=0.633333,
                wxlimt=24.00,
            ),
        ),
        # Oct. 16 through May
        WeatherFilterParamsForSeason(
            start_month=10,
            start_day=16,
            end_month=5,
            weather_filter_params=WeatherFilterParams(
                wintrc=53.4153,
                wslope=0.661017,
                wxlimt=22.00,
            ),
        ),
    ]
)
