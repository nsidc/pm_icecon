"""amsr2_goddard.py: bootstrap parameters file.

Bootstrap parameters derived by Goddard for use with AMSR2 from AU_SI products

Parameters were originally pulled from `ret_parameters_amsru2.f`.
"""
import datetime as dt

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
from pm_icecon.fetch.au_si import AU_SI_RESOLUTIONS
from pm_icecon.masks import get_ps_land_mask, get_ps_pole_hole_mask

AMSR2_NORTH_PARAMS = ParamsDict(
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

AMSR2_SOUTH_PARAMS = ParamsDict(
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
