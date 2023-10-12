"""AMSR2 Bootstrap parameters.

Bootstrap parameters for use with AMSR2 derived from AU_SI products

Parameters are based on values rom `ret_parameters_amsru2.f`. Updates have been
made to the weather filter paramters (`wxlimt`) for the CDR.
"""
import datetime as dt

from pm_tb_data.fetch.au_si import AU_SI_RESOLUTIONS

from pm_icecon._types import Hemisphere
from pm_icecon.bt._types import Line, Tiepoint
from pm_icecon.bt.masks import get_ps_invalid_ice_mask
from pm_icecon.bt.params._types import ParamsDict
from pm_icecon.config.models.bt import (
    BootstrapParams,
    TbSetParams,
    WeatherFilterParams,
    WeatherFilterParamsForSeason,
)
from pm_icecon.masks import get_ps_land_mask, get_ps_pole_hole_mask

GODDARD_AMSR2_NORTH_PARAMS = ParamsDict(
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

GODDARD_AMSR2_SOUTH_PARAMS = ParamsDict(
    vh37_params=TbSetParams(
        water_tie_point_set=(Tiepoint(207.6), Tiepoint(131.9)),
        ice_tie_point_set=(Tiepoint(259.4), Tiepoint(247.3)),
        lnline=Line(offset=-90.62, slope=1.2759),
    ),
    v1937_params=TbSetParams(
        water_tie_point_set=(Tiepoint(207.6), Tiepoint(182.7)),
        ice_tie_point_set=(Tiepoint(259.4), Tiepoint(261.6)),
        lnline=Line(offset=62.89, slope=0.7618),
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

# TODO: rename to indicate these are calculated for the CDR.
CDR_AMSR2_NORTH_PARAMS = GODDARD_AMSR2_NORTH_PARAMS.copy()
CDR_AMSR2_NORTH_PARAMS['weather_filter_seasons'] = [
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

CDR_AMSR2_SOUTH_PARAMS = GODDARD_AMSR2_SOUTH_PARAMS.copy()
CDR_AMSR2_SOUTH_PARAMS['weather_filter_seasons'] = [
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


# used to get parameters in `cdr.py`. Not used by the ecdr.
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
        **(CDR_AMSR2_NORTH_PARAMS if hemisphere == 'north' else CDR_AMSR2_SOUTH_PARAMS),
    )

    return bt_params