from typing import Optional

import numpy as np
import numpy.typing as npt

from cdr_amsr2._types import ValidSatellites
from cdr_amsr2.bt._types import Tiepoint
from cdr_amsr2.config.models.base_model import ConfigBaseModel


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

    water_tie_point: Tiepoint
    """Starting or 'default' water tie point (wtp) for this Tb set.

    A new wtp is calculated and used if the calculated wtp is within +/- 10 of
    the given wtp.
    """

    ice_tie_point: Tiepoint
    """Ice tie point (itp) for this Tb set.

    Used to calculate the coefficients `radslp` `radoff` `radlen`. See
    `calc_rad_coeffs_32`.
    """

    lnline: tuple[float, float]


class BootstrapParams(ConfigBaseModel):
    # TODO: what do these values represent? Are they likely to change from 0 and
    # -2?
    add1: float = 0.0
    add2: float = -2.0

    # Flags
    landval: float = 120.0
    """Flag value for cells representing land."""
    missval: float = 110.0
    """Flag value for cells representing missing data."""

    # TODO: what do these represent?
    minic: float = 10.0
    maxic: float = 1.0

    # min/max tb range. Any tbs outside this range are masked.
    mintb: float = 10.0
    """The minimum valid brightness temperature value."""
    maxtb: float = 320.0
    """The maximum valid brightness temperature value."""

    # TODO: do we really need minval/maxval specified? See note in
    # `bt.compute_bt_ic.fix_output_gdprod`.
    minval: float = 0
    """The minimum valid sea ice concentration."""
    maxval: float = 100
    """The maximum valid sea ice concentration."""

    # TODO: consider just adding this as an argument to the bootstrap alg
    # entrypoint?
    sat: ValidSatellites
    """String representing satellite."""

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