from cdr_amsr2.config.models.bt import (
    ParaNSB2,
    WeatherFilterParams,
    WeatherFilterParamsForSeason,
)

A2L1C_NORTH_PARAMS = ParaNSB2(
    weather_filter_seasons=[
        # June through October 15
        WeatherFilterParamsForSeason(
            start_month=6,
            end_month=10,
            end_day=15,
            weather_filter_params=WeatherFilterParams(
                wintrc=82.71,
                wslope=0.5352,
                wxlimt=23.34,
            ),
        ),
        # Oct. 16 through May
        WeatherFilterParamsForSeason(
            start_month=10,
            start_day=16,
            end_month=5,
            weather_filter_params=WeatherFilterParams(
                wintrc=84.73,
                wslope=0.5352,
                wxlimt=18.39,
            ),
        ),
    ]
)
