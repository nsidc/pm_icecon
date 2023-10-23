import numpy as np
import numpy.typing as npt
from scipy import ndimage


def spatial_interp_tbs(tbs_array: npt.NDArray):  # noqa
    """Perform spatial interpolation on input tbs field.

    Originally used by and defined for the nasateam algorithm.

    Attempts to interpolate Tbs that are `np.nan` or less than or equal to 0.
    """
    # Implement spatial interpolation scheme of SpatialInt_np.c
    # and SpatialInt_sp.c
    # Weighting scheme is: orthogonally adjacent weighted 1.0
    #                      diagonally adjacent weighted 0.707
    orig = tbs_array.copy()
    total = np.zeros_like(orig, dtype=np.float32)
    count = np.zeros_like(orig, dtype=np.float32)

    # NaN values do not work with `ndimage.shift`, so set them to 0.
    orig[np.isnan(orig)] = 0

    interp_locs = orig <= 0

    # Return the original array if there's nothing to interpolate.
    if not np.any(interp_locs):
        return tbs_array.copy()

    for offset in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        rolled = ndimage.shift(orig, offset, mode="nearest", order=0)
        has_vals = (rolled > 0) & (interp_locs)
        total[has_vals] += rolled[has_vals]
        count[has_vals] += 1.0

    for offset in ((1, 1), (1, -1), (-1, -1), (-1, 1)):
        rolled = ndimage.shift(orig, offset, mode="nearest", order=0)
        has_vals = (rolled > 0) & (interp_locs)
        total[has_vals] += 0.707 * rolled[has_vals]
        count[has_vals] += 0.707

    replace_locs = interp_locs & (count > 1.2)
    count[count == 0] = 1
    average = np.divide(total, count, dtype=np.float32)

    interp = tbs_array.copy()
    interp[replace_locs] = average[replace_locs]

    return interp
