"""Routines which implement land spillover corrections to sea ice concentration fields.

land_spillover.py
"""

from pathlib import Path

import numpy as np
import numpy.typing as npt
from loguru import logger
from scipy.signal import convolve2d

from pm_icecon.constants import NASATEAM2_ANCILLARY_DIR

# TODO: The various directory vars, eg anc_dir, should be either abstracted
#   as a constant or passed as a configuration parameter.

# TODO: So too, the filename template strings should be authoritatively
#   set in a central location.

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


def read_adj123_file(
    gridid: str = 'psn12.5',
    xdim: int = 608,
    ydim: int = 896,
    anc_dir: Path = NASATEAM2_ANCILLARY_DIR,
    adj123_fn_template: str = '{anc_dir}/coastal_adj_diag123_{gridid}.dat',
):
    """Read the diagonal adjacency 123 file."""
    coast_adj_filepath = Path(adj123_fn_template.format(anc_dir=anc_dir, gridid=gridid))
    assert coast_adj_filepath.is_file()
    adj123 = np.fromfile(coast_adj_filepath, dtype=np.uint8).reshape(ydim, xdim)

    return adj123


def create_land90_conc_file(
    gridid: str = 'psn12.5',
    xdim: int = 608,
    ydim: int = 896,
    anc_dir: Path = NASATEAM2_ANCILLARY_DIR,
    adj123_fn_template: str = '{anc_dir}/coastal_adj_diag123_{gridid}.dat',
    write_l90c_file: bool = True,
    l90c_fn_template: str = '{anc_dir}/land90_conc_{gridid}.dat',
):
    """Create the land90-conc file.

    The 'land90' array is a mock sea ice concentration array that is calculated
    from the land mask.  It assumes that the mock concentration value will be
    the average of a 7x7 array of local surface mask values centered on the
    center pixel.  Water grid cells are considered to have a sea ice
    concentration of zero.  Land grid cells are considered to have a sea ice
    concentration of 90%.  The average of the 49 grid cells in the 7x7 array
    yields the `land90` concentration value.
    """
    adj123 = read_adj123_file(gridid, xdim, ydim, anc_dir, adj123_fn_template)
    is_land = adj123 == 0
    is_coast = (adj123 == 1) | (adj123 == 2)

    ones_7x7 = np.ones((7, 7), dtype=np.uint8)

    land_count = convolve2d(is_land, ones_7x7, mode='same', boundary='symm')
    land90 = (land_count * 0.9) / 49 * 100.0
    land90 = land90.astype(np.float32)

    land90[~is_coast] = 0

    if write_l90c_file:
        l90c_fn = l90c_fn_template.format(anc_dir=anc_dir, gridid=gridid)
        land90.tofile(l90c_fn)
        print(f'Wrote: {l90c_fn}\n  {land90.dtype}  {land90.shape}')

    return land90


def load_or_create_land90_conc(
    gridid: str = 'psn12.5',
    xdim: int = 608,
    ydim: int = 896,
    anc_dir: Path = NASATEAM2_ANCILLARY_DIR,
    l90c_fn_template: str = '{anc_dir}/land90_conc_{gridid}.dat',
    overwrite: bool = False,
):
    # Attempt to load the land90_conc field, and if fail, create it
    l90c_filepath = Path(l90c_fn_template.format(anc_dir=anc_dir, gridid=gridid))
    if overwrite or not l90c_filepath.is_file():
        data = create_land90_conc_file(
            gridid, xdim, ydim, anc_dir=anc_dir, l90c_fn_template=l90c_fn_template
        )
    else:
        data = np.fromfile(l90c_filepath, dtype=np.float32).reshape(ydim, xdim)
        logger.info(f'Read NT2 land90 mask from:\n  {l90c_filepath}')

    return data


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
