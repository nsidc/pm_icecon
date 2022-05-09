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

data1 = np.fromfile(fn1, dtype=np.float32)
data2 = np.fromfile(fn2, dtype=np.float32)

diff = data2 - data1
print(f'diff min: {np.min(diff)}')
print(f'diff max: {np.max(diff)}')

if np.max(np.abs(diff)) < 0.0001:
    print('  GOOD!')
else:
    print("  oooh, is that okay...?")
