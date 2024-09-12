"""Routines which implement land spillover corrections to sea ice concentration fields.

In these NT2 algorithms:
    adj123 refers to an array that contains the "adjacency-to-land" values
    where:
        "0" are land grid cells.
        "1" are ocean grid cells adjacent to land
        "2" are ocean grid cells adjacent to "1" cells
        "3" are ocean grid cells adjacent to "2" cells

The algorithm has two passes:
    nt2a: Sets a grid cell's sea ice concentration to zero if all of the
          neighboring 3-from-land grid cells have zero percent conc
    nt2b: Sets a grid cell's sea ice concentration to zero if the `land90`
          mock concentration value is greater than the calculated conc value
"""

import numpy as np
import numpy.typing as npt
from scipy.signal import convolve2d


def apply_nt2a_land_spillover(
    conc: npt.NDArray,
    adj123: npt.NDArray,
    anchoring_siconc: float = 0.0,
    affect_dist3: bool = False,
) -> npt.NDArray:
    """Apply the first part of the NASA Team 2 land spillover routine.

    If all of the nearby 3-from-shore pixels have zero percent concentration,
    then so does this pixel

    anchoring_siconc refers to the minimum sea ice conc that will count as
    sufficient for anchoring near-shore values

    affect_dist3 determines whetheror not to affect pixels 3 away from shore.
    By default, only pixels up to 2 away from shore are affected

    Note: here, conc is in %, so ranges from 0-100% (or more, if not yet clamped)
    """
    if affect_dist3:
        is_modifiable = (adj123 == 1) | (adj123 == 2) | (adj123 == 3)
    else:
        is_modifiable = (adj123 == 1) | (adj123 == 2)
    is_adj3 = adj123 == 3

    is_zero_conc = conc <= anchoring_siconc

    # TODO: this should be extracted as a kwarg. 254 may not always be
    # consistently used as a flag value for land!
    is_land = conc == 254

    kernel = np.ones((7, 7), dtype=np.uint8)

    ones_adj3 = np.zeros(conc.shape, dtype=np.uint8)
    ones_adj3[is_adj3] = 1
    n_near_adj3 = convolve2d(ones_adj3, kernel, mode="same", boundary="symm")

    ones_adj3_zero = np.zeros(conc.shape, dtype=np.uint8)
    ones_adj3_zero[is_adj3 & is_zero_conc] = 1
    n_near_adj3_zeros = convolve2d(ones_adj3_zero, kernel, mode="same", boundary="symm")

    conc[
        (n_near_adj3_zeros > 0)
        & (n_near_adj3_zeros == n_near_adj3)
        & (is_modifiable)
        & (~is_land)
    ] = 0

    return conc


def create_land90(*, adj123: npt.NDArray) -> npt.NDArray:
    """Create and return l90c from the provided `adj123`.

    The 'land90' array is a mock sea ice concentration array that is calculated
    from the land mask.  It assumes that the mock concentration value will be
    the average of a 7x7 array of local surface mask values centered on the
    center pixel.  Water grid cells are considered to have a sea ice
    concentration of zero.  Land grid cells are considered to have a sea ice
    concentration of 90%.  The average of the 49 grid cells in the 7x7 array
    yields the `land90` concentration value.
    """
    is_land = adj123 == 0
    is_coast = (adj123 == 1) | (adj123 == 2)

    ones_7x7 = np.ones((7, 7), dtype=np.uint8)

    land_count = convolve2d(is_land, ones_7x7, mode="same", boundary="symm")
    land90 = (land_count * 0.9) / 49 * 100.0
    land90 = land90.astype(np.float32)

    land90[~is_coast] = 0

    return land90


def apply_nt2b_land_spillover(
    conc: npt.NDArray,
    adj123: npt.NDArray,
    l90c: npt.NDArray,
) -> npt.NDArray:
    """Apply the second part of the NASA Team 2 land spillover routine.

    If the calculated concentration is less than the 7x7 box land-is-90%
    average conc value, then set it to zero
    """
    l90c_ge_conc = l90c >= conc
    conc[l90c_ge_conc] = 0

    return conc


def apply_nt2_land_spillover(
    *,
    conc: npt.NDArray,
    adj123: npt.NDArray,
    l90c: npt.NDArray,
    anchoring_siconc: float = 0.0,
    affect_dist3: bool = False,
) -> npt.NDArray:
    """Apply first and second passes of NASA Team 2 land spillover routine.

    Note: here, conc is in %, so ranges from 0-100% (or more, if not yet clamped)"""
    spillover_applied_conc0 = conc.copy()
    spillover_applied_conca = apply_nt2a_land_spillover(
        spillover_applied_conc0,
        adj123,
        anchoring_siconc=anchoring_siconc,
        affect_dist3=affect_dist3,
    )
    spillover_applied_concb = apply_nt2b_land_spillover(
        spillover_applied_conca, adj123, l90c
    )

    return spillover_applied_concb
