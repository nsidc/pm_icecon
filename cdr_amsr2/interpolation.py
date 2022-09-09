from typing import Optional

import numpy.typing as npt
import numpy as np


def spatial_interp_conc(  # noqa
    sat,  # TODO: type of 'sat'
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

    # Now, replace pole if e2n6.25
    if sat == 'a2l1c':
        # TODO: This pole hole function needs some work(!)
        print('Setting pole hole for a2l1c')

        iceout_nearpole = iceout[820:860, 820:860]

        is_pole = iceout_nearpole == 0

        iceout_nearpole[is_pole] = 110

        print(f'Replaced {np.sum(np.where(is_pole, 1, 0))} values at pole')

    return iceout


def spatial_interp_tbs(tbs):  # noqa
    """Perform spatial interpolation on input tbs.

    Originally used by and defined for the nasateam algorithm.
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

        interp_locs = orig <= 0

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
