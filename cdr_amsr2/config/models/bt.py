from typing import Optional

import numpy as np
import numpy.typing as npt

from cdr_amsr2._types import ValidSatellites
from cdr_amsr2.config.models.base_model import ConfigBaseModel


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

    # TODO: what do these values represent?
    # TODO: enforce length of list w/ pydantic validator.
    # TODO: tuple?
    # TODO: do we want these values as params? they get overwritten by `ret_para_nsb2`
    # ln1: list[float]  # len 2
    # ln2: list[float]  # len 2

    # TODO: in the code, this is actually given by `ret_para_nsb2`. Should we
    # let it be overridden? What does this represent/do?
    # lnchk: float = 1.5

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

    # TODO: rename to 'season'. Can we just extract this from the date?
    # What does 1 even mean?
    seas: int = 1
    """Season."""

    # TODO: change to boolean type mask
    land_mask: npt.NDArray[np.int16]

    # TODO: change to boolean type mask
    # Hemisphere dependent. Should be required for Northern hemisphere. Should
    # be exlcluded in South. TODO: should we create a custom validator for this?
    # We would also want to add Hemisphere to this config object as well in that
    # case.
    pole_mask: Optional[npt.NDArray[np.int16]] = None
