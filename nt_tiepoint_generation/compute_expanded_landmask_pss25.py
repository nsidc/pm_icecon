"""
compute_expanded_landmask_pss25.py

Use the PSS25km land mask as a basis,
then mask out an additional 3 grid cells from land
"""

import numpy as np
from scipy.signal import convolve2d


xdim = 316
ydim = 332

ifn = './pss25_loili.dat'
smask_init = np.fromfile(ifn, dtype=np.uint8).reshape(ydim, xdim)
if 'loili' in ifn:
    smask_init[smask_init == 50] = 0  # Convert loili to 0 = ocean
smask_init[smask_init != 0] = 1  # this is for "landmask" file

kernel = np.array(
    [[0, 0, 1, 1, 1, 0, 0],
     [0, 1, 1, 1, 1, 1, 0],
     [1, 1, 1, 1, 1, 1, 1],
     [1, 1, 1, 1, 1, 1, 1],
     [1, 1, 1, 1, 1, 1, 1],
     [0, 1, 1, 1, 1, 1, 0],
     [0, 0, 1, 1, 1, 0, 0]]
)

print(f'Expanding landmask with kernel:\n{kernel}')

smask_convolved = convolve2d(
    smask_init,
    kernel,
    mode='same',
    boundary='symm').astype(np.uint8)

smask_exp = np.zeros((ydim, xdim), dtype=np.uint8)
smask_exp[smask_convolved != 0] = 100

ofn = './pss25_expanded_landmask.dat'
smask_exp.tofile(ofn)
print(f'Wrote: {ofn}  {smask_exp.dtype}  {smask_exp.shape}')
