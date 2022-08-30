"""
create_doymasks.py

Starting with raw day-of-year if-ever valid ice masks,
create 3-day rolling if-ever masks
"""

import os

import numpy as np

# NH
n_ifn_ = './bt_doymasks_raw/bt_doymask_n_{doy:03d}.dat'

n_outdir = './bt_doymasks_nh'
os.makedirs(n_outdir, exist_ok=True)
n_ofn_ = '{n_outdir}/bt_doymask_n_{doy:03d}.dat'

xdim = 304
ydim = 448

# Set 365 and 366 to each other
mask365 = np.fromfile(n_ifn_.format(doy=365), dtype=np.int8).reshape(ydim, xdim)
mask366 = np.fromfile(n_ifn_.format(doy=366), dtype=np.int8).reshape(ydim, xdim)
is_mask_lastday = (mask365 > 0) | (mask366 > 0)
mask365[is_mask_lastday] = 100
mask366[is_mask_lastday] = 100
mask365.tofile(n_ifn_.format(doy=365))
mask366.tofile(n_ifn_.format(doy=366))

# Just do the first and last doy by hand
doy = 1
prior = np.fromfile(n_ifn_.format(doy=365), dtype=np.uint8).reshape(ydim, xdim)
this = np.fromfile(n_ifn_.format(doy=1), dtype=np.uint8).reshape(ydim, xdim)
nextd = np.fromfile(n_ifn_.format(doy=2), dtype=np.uint8).reshape(ydim, xdim)
mask = np.zeros((ydim, xdim), dtype=np.uint8)
is_prior = prior != 0
is_this = this != 0
is_next = nextd != 0
mask[is_prior | is_this | is_next] = 100
ofn = n_ofn_.format(n_outdir=n_outdir, doy=1)
mask.tofile(ofn)
print(f'Wrote: {ofn}')

for doy in range(2, 365):
    prior = np.fromfile(n_ifn_.format(doy=doy - 1), dtype=np.uint8).reshape(ydim, xdim)
    this = np.fromfile(n_ifn_.format(doy=doy), dtype=np.uint8).reshape(ydim, xdim)
    nextd = np.fromfile(n_ifn_.format(doy=doy + 1), dtype=np.uint8).reshape(ydim, xdim)
    mask = np.zeros((ydim, xdim), dtype=np.uint8)
    is_prior = prior != 0
    is_this = this != 0
    is_next = nextd != 0
    mask[is_prior | is_this | is_next] = 100
    ofn = n_ofn_.format(n_outdir=n_outdir, doy=doy)
    mask.tofile(ofn)
    print(f'Wrote: {ofn}')

doy = 365
prior = np.fromfile(n_ifn_.format(doy=364), dtype=np.uint8).reshape(ydim, xdim)
this = np.fromfile(n_ifn_.format(doy=365), dtype=np.uint8).reshape(ydim, xdim)
nextd = np.fromfile(n_ifn_.format(doy=1), dtype=np.uint8).reshape(ydim, xdim)
mask = np.zeros((ydim, xdim), dtype=np.uint8)
is_prior = prior != 0
is_this = this != 0
is_next = nextd != 0
mask[is_prior | is_this | is_next] = 100
ofn = n_ofn_.format(n_outdir=n_outdir, doy=365)
mask.tofile(ofn)
print(f'Wrote: {ofn}')

# SH
s_ifn_ = './bt_doymasks_raw/bt_doymask_s_{doy:03d}.dat'

s_outdir = './bt_doymasks_sh'
os.makedirs(s_outdir, exist_ok=True)
s_ofn_ = '{s_outdir}/bt_doymask_s_{doy:03d}.dat'

# NH
xdim = 316
ydim = 332

# Set 365 and 366 to each other
mask365 = np.fromfile(s_ifn_.format(doy=365), dtype=np.int8).reshape(ydim, xdim)
mask366 = np.fromfile(s_ifn_.format(doy=366), dtype=np.int8).reshape(ydim, xdim)
is_mask_lastday = (mask365 > 0) | (mask366 > 0)
mask365[is_mask_lastday] = 100
mask366[is_mask_lastday] = 100
mask365.tofile(s_ifn_.format(doy=365))
mask366.tofile(s_ifn_.format(doy=366))

# Just do the first and last doy by hand
doy = 1
prior = np.fromfile(s_ifn_.format(doy=365), dtype=np.uint8).reshape(ydim, xdim)
this = np.fromfile(s_ifn_.format(doy=1), dtype=np.uint8).reshape(ydim, xdim)
nextd = np.fromfile(s_ifn_.format(doy=2), dtype=np.uint8).reshape(ydim, xdim)
mask = np.zeros((ydim, xdim), dtype=np.uint8)
is_prior = prior != 0
is_this = this != 0
is_next = nextd != 0
mask[is_prior | is_this | is_next] = 100
ofn = s_ofn_.format(s_outdir=s_outdir, doy=1)
mask.tofile(ofn)
print(f'Wrote: {ofn}')

for doy in range(2, 365):
    prior = np.fromfile(s_ifn_.format(doy=doy - 1), dtype=np.uint8).reshape(ydim, xdim)
    this = np.fromfile(s_ifn_.format(doy=doy), dtype=np.uint8).reshape(ydim, xdim)
    nextd = np.fromfile(s_ifn_.format(doy=doy + 1), dtype=np.uint8).reshape(ydim, xdim)
    mask = np.zeros((ydim, xdim), dtype=np.uint8)
    is_prior = prior != 0
    is_this = this != 0
    is_next = nextd != 0
    mask[is_prior | is_this | is_next] = 100
    ofn = s_ofn_.format(s_outdir=s_outdir, doy=doy)
    mask.tofile(ofn)
    print(f'Wrote: {ofn}')

doy = 365
prior = np.fromfile(s_ifn_.format(doy=364), dtype=np.uint8).reshape(ydim, xdim)
this = np.fromfile(s_ifn_.format(doy=365), dtype=np.uint8).reshape(ydim, xdim)
nextd = np.fromfile(s_ifn_.format(doy=1), dtype=np.uint8).reshape(ydim, xdim)
mask = np.zeros((ydim, xdim), dtype=np.uint8)
is_prior = prior != 0
is_this = this != 0
is_next = nextd != 0
mask[is_prior | is_this | is_next] = 100
ofn = s_ofn_.format(s_outdir=s_outdir, doy=365)
mask.tofile(ofn)
print(f'Wrote: {ofn}')
