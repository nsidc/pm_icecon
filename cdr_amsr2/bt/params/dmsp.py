"""Parameters for DMSP satellites.

Defense Meteorological Satellite Program. E.g, f17, f18.
"""
from cdr_amsr2.config.models.bt import (
    ParaNSB2,
    TbSetParams,
    WeatherFilterParams,
    WeatherFilterParamsForSeason,
)

DMSP_vh37_params = TbSetParams(
    water_tie_point=[201.916, 132.815],
    ice_tie_point=[255.670, 241.713],
    lnline=[-73.5471, 1.21104],
    iceline=[-25.9729, 1.04382],
)
DMSP_v1937_params = TbSetParams(
    water_tie_point=[201.916, 178.771],
    ice_tie_point=[255.670, 258.341],
    lnline=[47.0061, 0.809335],
    iceline=[112.803, 0.550296],
)

F17_F18_NORTH_PARAMS = ParaNSB2(
    vh37_params=DMSP_vh37_params,
    v1937_params=DMSP_v1937_params,
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
    ],
)
