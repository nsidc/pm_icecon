"""Parameter from Goddard for CLASS data.

CLASS is the NOAA Comprehensive Large Array-Data Stewardship System. Data from
CLASS is used for the NRT CDR (g10016).

The parameters defined in this file are not suitable for use with data from
Remote Sensing Systems (RSS). Data from RSS is used for the 'final' CDR (g02202).

All parameters pulled from `ret_parameters_sb2.f`.

TODO: separate out params into sat-specific modules.

TODO: bootstrap parameters are being refactored. See the `nsidc0001` bt params
module for the most up-to-date method/approach.
"""

from pm_icecon.bt._types import Line, Tiepoint
from pm_icecon.config.models.bt import (
    TbSetParams,
    WeatherFilterParams,
    WeatherFilterParamsForSeason,
)

north_vh37_params = TbSetParams(
    water_tie_point_set=(Tiepoint(201.916), Tiepoint(132.815)),
    ice_tie_point_set=(Tiepoint(255.670), Tiepoint(241.713)),
    lnline=Line(offset=-73.5471, slope=1.21104),
)
north_v1937_params = TbSetParams(
    water_tie_point_set=(Tiepoint(201.916), Tiepoint(178.771)),
    ice_tie_point_set=(Tiepoint(255.670), Tiepoint(258.341)),
    lnline=Line(offset=47.0061, slope=0.809335),
)

south_vh37_params = TbSetParams(
    water_tie_point_set=(Tiepoint(201.990), Tiepoint(133.943)),
    ice_tie_point_set=(Tiepoint(259.122), Tiepoint(248.284)),
    lnline=Line(offset=-90.9384, slope=1.28239),
)
south_v1937_params = TbSetParams(
    water_tie_point_set=(Tiepoint(201.990), Tiepoint(178.358)),
    ice_tie_point_set=(Tiepoint(259.122), Tiepoint(261.654)),
    lnline=Line(offset=61.7438, slope=0.767205),
)

# NOTE: The Goddard code specifically indicated that these parameters were for
# F17 and F18. However, we think that because F16 is a SSMIS instrument, these
# parameters should work for it as well.
SSMIS_NORTH_PARAMS = dict(
    vh37_params=north_vh37_params,
    v1937_params=north_v1937_params,
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

# NOTE: The Goddard code specifically indicated that these parameters were for
# F17 and F18. However, we think that because F16 is a SSMIS instrument, these
# parameters should work for it as well.
SSMIS_SOUTH_PARAMS = dict(  # noqa
    vh37_params=south_vh37_params,
    v1937_params=south_v1937_params,
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


SMMR_NORTH_PARAMS = dict(
    vh37_params=north_vh37_params,
    v1937_params=north_v1937_params,
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
    ],
)

SMRR_SOUTH_PARAMS = dict(  # noqa
    # TODO: should we be using these 'DMSP' param values for SMMR? This is how
    # the code in `ret_para_nsb2` used to work.
    vh37_params=south_vh37_params,
    v1937_params=south_v1937_params,
    weather_filter_seasons=[
        # Just one season for the Southern hemisphere.
        WeatherFilterParamsForSeason(
            start_month=1,
            end_month=12,
            weather_filter_params=WeatherFilterParams(
                wintrc=82.5000,
                wslope=0.529236,
                wxlimt=24.82,
            ),
        ),
    ],
)


# NOTE: use these parameters with caution! The code provided by Goddard
# indicates that a `sat` other than F17 or F18 should take these parameters but
# it isn't clear which other `sats` these were intended for. We think these are
# appropriate for SSM/I (F08, F10, F11, F13, F14, F15).
OTHER_NORTH_PARAMS = dict(  # noqa
    vh37_params=north_vh37_params,
    v1937_params=north_v1937_params,
    weather_filter_seasons=[
        # June through October 15
        WeatherFilterParamsForSeason(
            start_month=6,
            end_month=10,
            end_day=15,
            weather_filter_params=WeatherFilterParams(
                wintrc=89.3316,
                wslope=0.501537,
                wxlimt=21.0,
            ),
        ),
        # Oct. 16 through May
        WeatherFilterParamsForSeason(
            start_month=10,
            start_day=16,
            end_month=5,
            weather_filter_params=WeatherFilterParams(
                wintrc=90.3355,
                wslope=0.501537,
                wxlimt=14.0,
            ),
        ),
    ],
)
