"""Parameters for DMSP satellites.

Defense Meteorological Satellite Program. E.g, f17, f18.

All parameters pulled from `ret_parameters_sb2.f`.
"""
from cdr_amsr2.config.models.bt import (
    TbSetParams,
    WeatherFilterParams,
    WeatherFilterParamsForSeason,
)

dmsp_north_vh37_params = TbSetParams(
    water_tie_point=[201.916, 132.815],
    ice_tie_point=[255.670, 241.713],
    lnline=[-73.5471, 1.21104],
)
dmsp_north_v1937_params = TbSetParams(
    water_tie_point=[201.916, 178.771],
    ice_tie_point=[255.670, 258.341],
    lnline=[47.0061, 0.809335],
)

dmsp_south_vh37_params = TbSetParams(
    water_tie_point=[201.990, 133.943],
    ice_tie_point=[259.122, 248.284],
    lnline=[-90.9384, 1.28239],
)
dmsp_south_v1937_params = TbSetParams(
    water_tie_point=[201.990, 178.358],
    ice_tie_point=[259.122, 261.654],
    lnline=[61.7438, 0.767205],
)

# NOTE: we think these are defined for NSIDC-0080 (derived from CLASS) and NOT
# from NSIDC-0001 (RSS)
# NOTE: the F17 and F18 CLASS params can also be used for F16 (as a starting
# point) because it is also an SSMIS instrument.
F17_F18_NORTH_PARAMS = dict(
    vh37_params=dmsp_north_vh37_params,
    v1937_params=dmsp_north_v1937_params,
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

F17_F18_SOUTH_PARAMS = dict(  # noqa
    vh37_params=dmsp_south_vh37_params,
    v1937_params=dmsp_south_v1937_params,
    weather_filter_seasons=[
        # Just one season for the Southern hemisphere.
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
