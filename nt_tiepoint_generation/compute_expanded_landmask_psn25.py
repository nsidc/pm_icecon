"""
compute_expanded_landmask_psn25.py

Use the PSN25km land mask as a basis,
then mask out an additional 3 grid cells from land
"""

import numpy as np
from scipy.signal import convolve2d

xdim = 304
ydim = 448

ifn = "./psn25_landmask.dat"
nmask_init = np.fromfile(ifn, dtype=np.uint8).reshape(ydim, xdim)
nmask_init[nmask_init != 0] = 1

kernel = np.array(
    [
        [0, 0, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 0, 0],
    ]
)

print(f"Expanding landmask with kernel:\n{kernel}")

nmask_convolved = convolve2d(nmask_init, kernel, mode="same", boundary="symm").astype(
    np.uint8
)

nmask_exp = np.zeros((ydim, xdim), dtype=np.uint8)
nmask_exp[nmask_convolved != 0] = 100

ofn = "./psn25_expanded_landmask.dat"
nmask_exp.tofile(ofn)
print(f"Wrote: {ofn}  {nmask_exp.dtype}  {nmask_exp.shape}")
