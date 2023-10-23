import datetime as dt

import numpy as np
import numpy.typing as npt

from pm_icecon._types import Hemisphere
from pm_icecon.constants import CDR_TESTDATA_DIR
from pm_icecon.util import get_ps25_grid_shape


def get_ps25_sst_mask(
    *,
    hemisphere: Hemisphere,
    date: dt.date,
    sst_threshold=2780,
) -> npt.NDArray[np.bool_]:
    """Read and return the ps25 SST mask.

    `True` elements are those which are masked as invalid ice.
    """
    # TODO: why are the northern hemisphere files 'fixed' while the southern
    # hemisphere are not (except in one case)?
    month_abbr = f"{date:%b}".lower()
    if hemisphere == "north":
        sst_fn = f"{month_abbr}.temp.zdf.ssmi_fixed_25fill.fixed"
    else:
        sst_fn = f"{month_abbr}.temp.zdf.ssmi_25fill"

    sst_path = (
        CDR_TESTDATA_DIR
        / f"nt_datafiles/data36/SST/{hemisphere.capitalize()}/"
        / sst_fn
    )
    sst_field = np.fromfile(sst_path, dtype=">i2")[150:].reshape(
        get_ps25_grid_shape(hemisphere=hemisphere)
    )

    where_sst_high = sst_field >= sst_threshold

    return where_sst_high
