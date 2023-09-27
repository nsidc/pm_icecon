from typing import Optional

import numpy as np
import numpy.typing as npt

from pm_icecon.bt._types import Line, Tiepoint, TiepointSet
from pm_icecon.config.models.base_model import ConfigBaseModel


def cast_as_TiepointSet(tp1, tp2) -> TiepointSet:
    # Cast a pair of floats as a TiepointSet.
    return (Tiepoint(tp1), Tiepoint(tp2))


class WeatherFilterParams(ConfigBaseModel):
    wintrc: float
    """Water intercept."""

    wslope: float
    """Water slope."""

    wxlimt: float
    """Weather filter limit."""


# Maybe call "SeasonalWeatherFilterParams"
class WeatherFilterParamsForSeason(ConfigBaseModel):
    """Weather filter parameters for a given 'season'.

    Start and end day/months define the 'season'.

    Parameters used by `ret_water_ssmi` to flag pixels as open water.

    TODO: consider a `name` attribute that defaults to a string repr of the date
    range but could be overriden with e.g., 'winter'.
    """

    # TODO: validators (day must be between 1-31, month must be between 1-12)
    # start and end days are optional to account for months w/ varying end day
    start_month: int
    start_day: int | None = None
    end_month: int
    end_day: int | None = None

    weather_filter_params: WeatherFilterParams


class TbSetParams(ConfigBaseModel):
    """Model for parameters related to a set of 2 Tbs.

    Bootstrap code currently expects one of two sets of Tbs: vh37 or v1937.
    """

    water_tie_point_set: TiepointSet
    """Starting or 'default' water tie point set for this Tb set.

    A new wtp is calculated and used if the calculated wtp is within +/- 10 of
    the given wtp.
    """

    ice_tie_point_set: TiepointSet
    """Ice tie point set (itp) for this Tb set.

    Used to calculate the coefficients `radslp` `radoff` `radlen`. See
    `calc_rad_coeffs_32`.
    """

    lnline: Line


class BootstrapParams(ConfigBaseModel):
    # TODO: what do these values represent? Are they likely to change from 0 and
    # -2?
    add1: float = 0.0
    add2: float = -2.0

    # TODO: unify the units (percentages or fractions, not on each!) for these
    # two paramters.
    # The minimum ice concentration as a percentage (10 == 10%)
    minic: float = 10.0
    # The maximum ice concentration as a fractional value (1 == 100%)
    maxic: float = 1.0

    # min/max tb range. Any tbs outside this range are masked.
    mintb: float = 10.0
    """The minimum valid brightness temperature value."""
    maxtb: float = 320.0
    """The maximum valid brightness temperature value."""

    # TODO: change to boolean type mask
    land_mask: npt.NDArray[np.bool_]

    # TODO: change to boolean type mask
    # Hemisphere dependent. Should be required for Northern hemisphere. Should
    # be exlcluded in South. TODO: should we create a custom validator for this?
    # We would also want to add Hemisphere to this config object as well in that
    # case.
    pole_mask: Optional[npt.NDArray[np.bool_]] = None

    # TODO: validators:
    #   * No overlap between seasons
    #   * If only 1 season, date range should be full range. Otherwise at least
    #     two?
    weather_filter_seasons: list[WeatherFilterParamsForSeason]
    """List of seasons with associated weather filter parameters.

    Note: if a season is not defined for a given date, the bootstrap code will
    linearly interpolate paramter values based on adjacent seasons.
    """

    vh37_params: TbSetParams
    v1937_params: TbSetParams

    invalid_ice_mask: npt.NDArray[np.bool_]
    """Mask representing areas that are invalid for sea ice."""
