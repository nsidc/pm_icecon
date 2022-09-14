from typing import Optional

import numpy as np
import numpy.typing as npt


# TODO: should we keep this function around?
def spatial_interp_conc(  # noqa
    ice: npt.NDArray[np.float32],  # TODO: conc?
    missval: float,
    landval: float,
    pole_mask: Optional[npt.NDArray[np.bool_]],
) -> npt.NDArray[np.float32]:
    """Perform spatial interpolation on a concentration field.

    Originally used by the bootstrap algorithm.
    """
    iceout = ice.copy()
    # implement fortran's spatial_interp() routine
    # Use -200 as a not-valid ocean sentinel value
    # so that it works with np.roll
    oceanvals = iceout.copy()

    total = np.zeros_like(oceanvals, dtype=np.float32)
    count = np.zeros_like(oceanvals, dtype=np.int32)
    for joff in range(-1, 2):
        for ioff in range(-1, 2):
            # TODO: consider using `scipy.ndimage.shift` instead of `np.roll`
            # here and elsewhere in the code.
            rolled = np.roll(oceanvals, (joff, ioff), axis=(1, 0))
            not_land_nor_miss = (rolled != landval) & (rolled != missval)
            total[not_land_nor_miss] += rolled[not_land_nor_miss]
            count[not_land_nor_miss] += 1

    count[count == 0] = 1
    replace_vals = total / count

    replace_locs = (oceanvals == missval) & (count >= 1)

    if pole_mask is not None:
        replace_locs = replace_locs & ~pole_mask

    iceout[replace_locs] = replace_vals[replace_locs]

    return iceout


def spatial_interp_tbs(tbs):  # noqa
    """Perform spatial interpolation on input tbs.

    Originally used by and defined for the nasateam algorithm.

    Attempts to interpolate Tbs that are `np.nan` or less than or equal to 0.
    """
    # Implement spatial interpolation scheme of SpatialInt_np.c
    # and SpatialInt_sp.c
    # Weighting scheme is: orthogonally adjacent weighted 1.0
    #                      diagonally adjacent weighted 0.707
    interp_tbs = {}
    for tb in tbs.keys():
        orig = tbs[tb].copy()
        total = np.zeros_like(orig, dtype=np.float32)
        count = np.zeros_like(orig, dtype=np.float32)

        interp_locs = np.isnan(orig) | (orig <= 0)

        # continue to the next tb field if there's nothing to interpolate.
        if not np.any(interp_locs):
            continue

        for offset in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            rolled = np.roll(orig, offset, axis=(0, 1))
            has_vals = (rolled > 0) & (interp_locs)
            total[has_vals] += rolled[has_vals]
            count[has_vals] += 1.0

        for offset in ((1, 1), (1, -1), (-1, -1), (-1, 1)):
            rolled = np.roll(orig, offset, axis=(0, 1))
            has_vals = (rolled > 0) & (interp_locs)
            total[has_vals] += 0.707 * rolled[has_vals]
            count[has_vals] += 0.707

        replace_locs = interp_locs & (count > 1.2)
        count[count == 0] = 1
        average = np.divide(total, count, dtype=np.float32)

        interp = orig.copy()
        interp[replace_locs] = average[replace_locs]

        interp_tbs[tb] = interp

    return interp_tbs
