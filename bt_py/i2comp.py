"""
fpcompt.py

"""

import numpy as np
import os
import sys


fn1 = sys.argv[1]
fn2 = sys.argv[2]

assert os.path.isfile(fn1)
assert os.path.isfile(fn2)

data1 = np.fromfile(fn1, dtype=np.int16)
data2 = np.fromfile(fn2, dtype=np.int16)

diff = data2 - data1
print(f'diff min: {np.min(diff)}')
print(f'diff max: {np.max(diff)}')
print(f'num diffs: {np.sum(np.where(diff != 0, 1, 0))}')

if np.max(np.abs(diff)) < 2:
    print('  GOOD!')
else:
    print("  oooh, is that okay...?")

diff = diff.reshape(448, 304)
print(f'{np.where(diff != 0)}')
