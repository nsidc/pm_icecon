from cdr_amsr2.config.models.bt import (
    ParaNSB2,
    WeatherFilterParams,
    WeatherFilterParamsForSeason,
)

AMSR2_NORTH_PARAMS = ParaNSB2(
    weather_filter_seasons=[
        # May (`seas=2` in `boot_ice_amsru2_np.f`)
        WeatherFilterParamsForSeason(
            start_month=5,
            end_month=5,
        ),
        # June through Sept. (`seas=3`)
        WeatherFilterParamsForSeason(
            start_month=6,
            end_month=9,
        ),
        # October (`seas=4`)
        WeatherFilterParamsForSeason(
            start_month=10,
            end_month=10,
        ),
        # So, 'season 1' is November through April...this spans a year. start
        # month can come before end month year-wise. This means we'll just ened
        # to have some logic to handle it...
    ]
)
