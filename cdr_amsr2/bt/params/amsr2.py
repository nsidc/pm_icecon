from cdr_amsr2.config.models.bt import (
    TbSetParams,
    WeatherFilterParams,
    WeatherFilterParamsForSeason,
)

AMSR2_NORTH_PARAMS = dict(
    vh37_params=TbSetParams(
        water_tie_point=[207.2, 131.9],
        ice_tie_point=[256.3, 241.2],
        lnline=[-71.99, 1.20],
    ),
    v1937_params=TbSetParams(
        water_tie_point=[207.2, 182.4],
        ice_tie_point=[256.3, 258.9],
        lnline=[48.26, 0.8048],
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
        water_tie_point=[207.6, 131.9],
        ice_tie_point=[259.4, 247.3],
        lnline=[-90.62, 1.2759],
    ),
    v1937_params=TbSetParams(
        water_tie_point=[207.6, 182.7],
        ice_tie_point=[259.4, 261.6],
        lnline=[62.89, 0.7618],
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
