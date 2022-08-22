"""Params for 'other' sats.

Use these parameters with caution! These parameters can be used if the `sat` is
not set to one of ['u2', '17', '18', '00', 'a2l1c']. However, it would be good
to confirm exactly where these parameters come from and for which sats they
should be used for.

TODO: look back over the Goddard code to see which sats these params are
actually used for.
"""
from cdr_amsr2.config.models.bt import (
    ParaNSB2,
    WeatherFilterParams,
    WeatherFilterParamsForSeason,
)

OTHER_NORTH_PARAMS = ParaNSB2(  # noqa
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
    ]
)
