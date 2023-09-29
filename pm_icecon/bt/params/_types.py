from typing import TypedDict

from pm_icecon.config.models.bt import TbSetParams, WeatherFilterParamsForSeason


class ParamsDict(TypedDict):
    vh37_params: TbSetParams
    v1937_params: TbSetParams
    weather_filter_seasons: list[WeatherFilterParamsForSeason]
