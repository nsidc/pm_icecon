"""Routine(s) to fill the Northern Hemisphere pole hole.

fill_polehole.py

In general, these will be grid and sensor dependent
"""

import numpy as np
import numpy.typing as npt
from scipy.ndimage import binary_dilation


# TODO: differentiate this from the function in `compute_bt_ic`
def fill_pole_hole(
    *, conc: npt.NDArray, near_pole_hole_mask: npt.NDArray[np.bool_]
) -> npt.NDArray:
    """Fill pole hole using the average of data found within the mask.

    Assumes that some data is available in and adjacent to the masked area.

    Missing areas are given by `np.nan`.
    """
    extended_nearpole_mask = binary_dilation(near_pole_hole_mask)

    # Fill zeros or NaNs near the pole
    is_vals_near_pole = extended_nearpole_mask & (conc > 0)
    is_missing_near_pole = extended_nearpole_mask & np.isnan(conc)
    mean_near_pole = np.nanmean(conc[is_vals_near_pole])
    conc[is_missing_near_pole] = mean_near_pole

    return conc
