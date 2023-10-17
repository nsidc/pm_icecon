"""Routine(s) to fill the Northern Hemisphere pole hole.

fill_polehole.py

In general, these will be grid and sensor dependent
"""

import numpy as np
import numpy.typing as npt


# TODO: differentiate this from the function in `compute_bt_ic`
def fill_pole_hole(
    *, conc: npt.NDArray, near_pole_hole_mask: npt.NDArray[np.bool_]
) -> npt.NDArray:
    """Fill pole hole using the average of data found within the mask.

    Assumes that some data is available in the masked area.
    """
    # Fill zeros or NaNs near the pole
    is_vals_near_pole = near_pole_hole_mask & (conc > 0)
    is_missing_near_pole = near_pole_hole_mask & ((conc == 0) | np.isnan(conc))
    mean_near_pole = np.mean(conc[is_vals_near_pole])
    conc[is_missing_near_pole] = mean_near_pole

    return conc
