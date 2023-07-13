"""
Bootstrap parameters for use with AMSR2 from new CETB product,
  i.e. either NSIDC-0763 or NSIDC-0630 version 2

Parameters were originally provided by 'ret_parameters_amsru2.f', except:
  - new wxlimt values were calculated by during Spring 2023 NSIDC
    investigation of weather effects for the NOAA CDR version 5 product
"""

from pm_icecon.bt._types import Line, Tiepoint
from pm_icecon.config.models.bt import (
    TbSetParams,
    WeatherFilterParams,
    WeatherFilterParamsForSeason,
)

A2L1C_NORTH_PARAMS = dict(
    vh37_params=TbSetParams(
        water_tie_point_set=(Tiepoint(207.2), Tiepoint(131.9)),
        ice_tie_point_set=(Tiepoint(256.3), Tiepoint(241.2)),
        lnline=Line(offset=-71.99, slope=1.20),
    ),
    v1937_params=TbSetParams(
        water_tie_point_set=(Tiepoint(207.2), Tiepoint(182.4)),
        ice_tie_point_set=(Tiepoint(256.3), Tiepoint(258.9)),
        lnline=Line(offset=48.26, slope=0.8048),
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
                # wxlimt=23.34,  # Original value
                wxlimt=21.7,  # Median of calc'd wx coefs
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
                # wxlimt=18.39,  # Original value
                wxlimt=13.7,  # Median of calc'd wx coefs
            ),
        ),
    ],
)
