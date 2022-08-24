from cdr_amsr2.config.models.bt import (
    TbSetParams,
    WeatherFilterParams,
    WeatherFilterParamsForSeason,
)

A2L1C_NORTH_PARAMS = dict(
    vh37_params=TbSetParams(
        water_tie_point=[207.2, 131.9],
        ice_tie_point=[256.3, 241.2],
        lnline=[-71.99, 1.20],
        iceline=[-30.26, 1.0564],
    ),
    v1937_params=TbSetParams(
        water_tie_point=[207.2, 182.4],
        ice_tie_point=[256.3, 258.9],
        lnline=[48.26, 0.8048],
        iceline=[110.03, 0.5759],
    ),
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
    ],
)
