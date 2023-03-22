"""
land_spillover.py

Routines which implement land spillover corrections to sea ice concentration fields
"""

import os
import numpy as np
from scipy.signal import convolve2d


def xwm(m='exiting in xwm()'):
    raise SystemExit(m)


def read_adj123_file(
    gridid='psn12.5',
    xdim=608,
    ydim=896,
    anc_dir = '/share/apps/amsr2-cdr/nasateam2_ancillary',
    adj123_fn_='{anc_dir}/coastal_adj_diag123_{gridid}.dat',
):
    """Read the diagonal adjacency 123 file """
    coast_adj_fn = adj123_fn_.format(anc_dir=anc_dir, gridid=gridid)
    assert os.path.isfile(coast_adj_fn)
    adj123 = np.fromfile(coast_adj_fn, dtype=np.uint8).reshape(ydim, xdim)

    return adj123

    
def create_land90_conc_file(
    gridid='psn12.5',
    xdim=608,
    ydim=896,
    anc_dir = '/share/apps/amsr2-cdr/nasateam2_ancillary',
    adj123_fn_='{anc_dir}/coastal_adj_diag123_{gridid}.dat',
    write_l90c_file=True,
    l90c_fn_='{anc_dir}/land90_conc_{gridid}.dat',
):
    """Create the land90-conc file"""
    adj123 = read_adj123_file(gridid, xdim, ydim, anc_dir, adj123_fn_)
    is_land = adj123 == 0
    is_coast = (adj123 == 1) | (adj123 == 2)

    ones_7x7 = np.ones((7, 7), dtype=np.uint8)

    land_count = convolve2d(is_land, ones_7x7, mode='same', boundary='symm')
    land90 = (land_count * 0.9) / 49 * 100.0
    land90 = land90.astype(np.float32)

    land90[~is_coast] = 0

    if write_l90c_file:
        l90c_fn = l90c_fn_.format(anc_dir=anc_dir, gridid=gridid)
        land90.tofile(l90c_fn)
        print(f'Wrote: {l90c_fn}\n  {land90.dtype}  {land90.shape}')

    return land90


def load_or_create_land90_conc(
    gridid='psn12.5',
    xdim=608,
    ydim=896,
    anc_dir = '/share/apps/amsr2-cdr/nasateam2_ancillary',
    l90c_fn_='{anc_dir}/land90_conc_{gridid}.dat',
    overwrite=False,
):

    # Attempt to load the land90_conc field, and if fail, create it
    l90c_fn = l90c_fn_.format(anc_dir=anc_dir, gridid=gridid)
    if overwrite or not os.path.isfile(l90c_fn):
        data = \
            create_land90_conc_file(
                gridid, xdim, ydim, anc_dir=anc_dir, l90c_fn_=l90c_fn_)
    else:
        data = np.fromfile(l90c_fn, dtype=np.float32).reshape(ydim, xdim)
        print(f'Read land90 mask from:\n  {l90c_fn}')
        print(f'  land90: {data.dtype}  {data.shape}')

    return data


def apply_nt2a_land_spillover(conc, adj123):
    """
    Apply the first part of the NASA Team 2 land spillover routine

    If all of the nearby 3-from-shore pixels have zero percent concentration,
    then so does this pixel
    """
    is_adj12 = (adj123 == 1) | (adj123 == 2)
    is_adj3 = adj123 == 3
    is_zero_conc = conc == 0.0
    # is_verylow_conc = conc <= 10.0
    is_land = conc == 254

    kernel = np.ones((7, 7), dtype=np.uint8)

    ones_adj3 = np.zeros(conc.shape, dtype=np.uint8)
    ones_adj3[is_adj3] = 1
    n_near_adj3 = convolve2d(ones_adj3, kernel,  mode='same', boundary='symm')

    ones_adj3_zero = np.zeros(conc.shape, dtype=np.uint8)
    ones_adj3_zero[is_adj3 & is_zero_conc] = 1
    # ones_adj3_zero[is_adj3 & is_verylow_conc] = 1
    n_near_adj3_zeros = convolve2d(ones_adj3_zero, kernel, mode='same', boundary='symm')

    conc[
        (n_near_adj3_zeros > 0) &
        (n_near_adj3_zeros == n_near_adj3) &
        (is_adj12) &
        (~is_land)
        ] = 0

    return conc


def apply_nt2b_land_spillover(conc, adj123, l90c):
    """
    Apply the second part of the NASA Team 2 land spillover routine

    If the calculated concentration is less than the 7x7 box land-is-90%
    average conc value, then set it to zero
    """
    is_adj12 = (adj123 == 1) | (adj123 == 2)
    is_adj3 = adj123 == 3
    is_land = conc == 254

    l90c_ge_conc = l90c >= conc
    conc[l90c_ge_conc] = 0

    return conc
