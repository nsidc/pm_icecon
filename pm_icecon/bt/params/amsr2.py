"""AMSR2 Bootstrap parameters.

Bootstrap parameters for use with AMSR2 derived from AU_SI products

Parameters are based on values rom `ret_parameters_amsru2.f`. Updates have been
made to the weather filter paramters (`wxlimt`).
"""
import datetime as dt

from pm_tb_data.fetch.au_si import AU_SI_RESOLUTIONS

from pm_icecon._types import Hemisphere
from pm_icecon.bt.masks import get_ps_invalid_ice_mask
from pm_icecon.bt.params.amsr2_goddard import AMSR2_NORTH_PARAMS as goddard_north_params
from pm_icecon.bt.params.amsr2_goddard import AMSR2_SOUTH_PARAMS as goddard_south_params
from pm_icecon.config.models.bt import (
    BootstrapParams,
    WeatherFilterParams,
    WeatherFilterParamsForSeason,
)
from pm_icecon.masks import get_ps_land_mask, get_ps_pole_hole_mask

AMSR2_NORTH_PARAMS = goddard_north_params.copy()
AMSR2_NORTH_PARAMS['weather_filter_seasons'] = [
    # November through April (`seas=1` in `boot_ice_amsru2_np.f`)
    WeatherFilterParamsForSeason(
        start_month=11,
        end_month=4,
        weather_filter_params=WeatherFilterParams(
            wintrc=84.73,
            wslope=0.5352,
            # The wxlimit was updated from 18.39
            wxlimt=13.7,
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
            # The wxlimt was updated from 23.34
            wxlimt=21.7,
        ),
    ),
    # October (`seas=4`) will get interpolated from the previous and next
    # (first in this list) season.
]

AMSR2_SOUTH_PARAMS = goddard_south_params.copy()
AMSR2_SOUTH_PARAMS['weather_filter_seasons'] = [
    # Just one season for the S. hemisphere.
    WeatherFilterParamsForSeason(
        start_month=1,
        end_month=12,
        weather_filter_params=WeatherFilterParams(
            wintrc=85.13,
            wslope=0.5379,
            # The wxlimit was updated from 18.596
            wxlimt=14.3,
        ),
    ),
]


def get_amsr2_params(
    *,
    date: dt.date,
    hemisphere: Hemisphere,
    resolution: AU_SI_RESOLUTIONS,
) -> BootstrapParams:
    invalid_ice_mask = get_ps_invalid_ice_mask(
        hemisphere=hemisphere,
        date=date,
        resolution=resolution,  # type: ignore[arg-type]
    )

    bt_params = BootstrapParams(
        land_mask=get_ps_land_mask(hemisphere=hemisphere, resolution=resolution),
        # There's no pole hole in the southern hemisphere.
        pole_mask=(
            get_ps_pole_hole_mask(resolution=resolution)
            if hemisphere == 'north'
            else None
        ),
        invalid_ice_mask=invalid_ice_mask,
        **(AMSR2_NORTH_PARAMS if hemisphere == 'north' else AMSR2_SOUTH_PARAMS),
    )

    return bt_params
