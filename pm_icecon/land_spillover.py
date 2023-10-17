"""Routines which implement land spillover corrections to sea ice concentration fields.

land_spillover.py
"""

import numpy as np
import numpy.typing as npt
from scipy.signal import convolve2d

"""
In these NT2 algorithms:
    adj123 refers to an array that contains the "adjacency-to-land" values
    where:
        "1" are ocean grid cells adjacent to land
        "2" are ocean grid cells adjacent to "1" cells
        "3" are ocean grid cells adjacent to "2" cells

The algorithm has two passes:
    nt2a: Sets a grid cell's sea ice concentration to zero if all of the
          neighboring 3-from-land grid cells have zero percent conc
    nt2b: Sets a grid cell's sea ice concentration to zero if the `land90`
          mock concentration value is greater than the calculated conc value
"""


def apply_nt2a_land_spillover(
    conc: npt.NDArray,
    adj123: npt.NDArray,
):
    """Apply the first part of the NASA Team 2 land spillover routine.

    If all of the nearby 3-from-shore pixels have zero percent concentration,
    then so does this pixel
    """
    is_adj12 = (adj123 == 1) | (adj123 == 2)
    is_adj3 = adj123 == 3
    is_zero_conc = conc == 0.0
    is_land = conc == 254

    kernel = np.ones((7, 7), dtype=np.uint8)

    ones_adj3 = np.zeros(conc.shape, dtype=np.uint8)
    ones_adj3[is_adj3] = 1
    n_near_adj3 = convolve2d(ones_adj3, kernel, mode='same', boundary='symm')

    ones_adj3_zero = np.zeros(conc.shape, dtype=np.uint8)
    ones_adj3_zero[is_adj3 & is_zero_conc] = 1
    n_near_adj3_zeros = convolve2d(ones_adj3_zero, kernel, mode='same', boundary='symm')

    conc[
        (n_near_adj3_zeros > 0)
        & (n_near_adj3_zeros == n_near_adj3)
        & (is_adj12)
        & (~is_land)
    ] = 0

    return conc


def apply_nt2b_land_spillover(
    conc: npt.NDArray,
    adj123: npt.NDArray,
    l90c: npt.NDArray,
):
    """Apply the second part of the NASA Team 2 land spillover routine.

    If the calculated concentration is less than the 7x7 box land-is-90%
    average conc value, then set it to zero
    """
    l90c_ge_conc = l90c >= conc
    conc[l90c_ge_conc] = 0

    return conc
