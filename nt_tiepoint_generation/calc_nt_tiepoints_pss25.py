"""
calc_nt_tiepoints_psn25.py

Using the tabulated daily slope and offset for the TB channels
compute the new NASA Team tie points

F17 tiepoint values are from:
    pmalgos repo
      seaice_goddard.f
"""

import os
import sys

import pandas as pd

# From seaice_goddard.f
# Note: no tiepoints for 22v (or 37v)
# Northern Hemisphere
tp_f17_n19h_ow = 113.4
tp_f17_n19v_ow = 184.9
tp_f17_n37v_ow = 207.1

tp_f17_n19h_fy = 232.0
tp_f17_n19v_fy = 248.4
tp_f17_n37v_fy = 242.3

tp_f17_n19h_my = 196.0
tp_f17_n19v_my = 220.7
tp_f17_n37v_my = 188.5

# Southern Hemisphere
tp_f17_s19h_ow = 113.4
tp_f17_s19v_ow = 184.9
tp_f17_s37v_ow = 207.1

tp_f17_s19h_fy = 237.8
tp_f17_s19v_fy = 253.1
tp_f17_s37v_fy = 246.6

tp_f17_s19h_my = 211.9
tp_f17_s19v_my = 244.0
tp_f17_s37v_my = 212.6

try:
    ifn = sys.argv[1]
    assert os.path.isfile(ifn)
except IndexError:
    raise SystemExit('No input file given')
except FileNotFoundError:
    raise SystemExit(f'Input file does not exist: {ifn}')

df = pd.read_fwf(ifn)

if '_nh' in ifn:
    tp_am2_n19h_ow = tp_f17_n19h_ow * df["n19h_m"].mean() + df["n19h_b"].mean()
    tp_am2_n19h_fy = tp_f17_n19h_fy * df["n19h_m"].mean() + df["n19h_b"].mean()
    tp_am2_n19h_my = tp_f17_n19h_my * df["n19h_m"].mean() + df["n19h_b"].mean()

    tp_am2_n19v_ow = tp_f17_n19v_ow * df["n19v_m"].mean() + df["n19v_b"].mean()
    tp_am2_n19v_fy = tp_f17_n19v_fy * df["n19v_m"].mean() + df["n19v_b"].mean()
    tp_am2_n19v_my = tp_f17_n19v_my * df["n19v_m"].mean() + df["n19v_b"].mean()

    tp_am2_n37v_ow = tp_f17_n37v_ow * df["n37v_m"].mean() + df["n37v_b"].mean()
    tp_am2_n37v_fy = tp_f17_n37v_fy * df["n37v_m"].mean() + df["n37v_b"].mean()
    tp_am2_n37v_my = tp_f17_n37v_my * df["n37v_m"].mean() + df["n37v_b"].mean()

    print('n19h:')
    print(f' slope mean: {df["n19h_m"].mean():9.5f}', end='')
    print(f'      std: {df["n19h_m"].std():9.5f}')
    print(f' intrc mean: {df["n19h_b"].mean():9.5f}', end='')
    print(f'      std: {df["n19h_b"].std():9.5f}')
    print('')
    print(f'    f17 tp n19h ow: {tp_f17_n19h_ow}')
    print(f'    am2 tp n19h ow: {tp_am2_n19h_ow:.2f}')
    print('')
    print(f'    f17 tp n19h fy: {tp_f17_n19h_fy}')
    print(f'    am2 tp n19h fy: {tp_am2_n19h_fy:.2f}')
    print('')
    print(f'    f17 tp n19h my: {tp_f17_n19h_my}')
    print(f'    am2 tp n19h my: {tp_am2_n19h_my:.2f}')
    print('')
    print('')
    print('n19v:')
    print(f' slope mean: {df["n19v_m"].mean():9.5f}', end='')
    print(f'      std: {df["n19v_m"].std():9.5f}')
    print(f' intrc mean: {df["n19v_b"].mean():9.5f}', end='')
    print(f'      std: {df["n19v_b"].std():9.5f}')
    print('')
    print(f'    f17 tp n19v ow: {tp_f17_n19v_ow}')
    print(f'    am2 tp n19v ow: {tp_am2_n19v_ow:.2f}')
    print('')
    print(f'    f17 tp n19v fy: {tp_f17_n19v_fy}')
    print(f'    am2 tp n19v fy: {tp_am2_n19v_fy:.2f}')
    print('')
    print(f'    f17 tp n19v my: {tp_f17_n19v_my}')
    print(f'    am2 tp n19v my: {tp_am2_n19v_my:.2f}')
    print('')
    print('')
    print('n37v:')
    print(f' slope mean: {df["n37v_m"].mean():9.5f}', end='')
    print(f'      std: {df["n37v_m"].std():9.5f}')
    print(f' intrc mean: {df["n37v_b"].mean():9.5f}', end='')
    print(f'      std: {df["n37v_b"].std():9.5f}')
    print('')
    print(f'    f17 tp n37v ow: {tp_f17_n37v_ow}')
    print(f'    am2 tp n37v ow: {tp_am2_n37v_ow:.2f}')
    print('')
    print(f'    f17 tp n37v fy: {tp_f17_n37v_fy}')
    print(f'    am2 tp n37v fy: {tp_am2_n37v_fy:.2f}')
    print('')
    print(f'    f17 tp n37v my: {tp_f17_n37v_my}')
    print(f'    am2 tp n37v my: {tp_am2_n37v_my:.2f}')
elif '_sh' in ifn:
    tp_am2_s19h_ow = tp_f17_s19h_ow * df["s19h_m"].mean() + df["s19h_b"].mean()
    tp_am2_s19h_fy = tp_f17_s19h_fy * df["s19h_m"].mean() + df["s19h_b"].mean()
    tp_am2_s19h_my = tp_f17_s19h_my * df["s19h_m"].mean() + df["s19h_b"].mean()

    tp_am2_s19v_ow = tp_f17_s19v_ow * df["s19v_m"].mean() + df["s19v_b"].mean()
    tp_am2_s19v_fy = tp_f17_s19v_fy * df["s19v_m"].mean() + df["s19v_b"].mean()
    tp_am2_s19v_my = tp_f17_s19v_my * df["s19v_m"].mean() + df["s19v_b"].mean()

    tp_am2_s37v_ow = tp_f17_s37v_ow * df["s37v_m"].mean() + df["s37v_b"].mean()
    tp_am2_s37v_fy = tp_f17_s37v_fy * df["s37v_m"].mean() + df["s37v_b"].mean()
    tp_am2_s37v_my = tp_f17_s37v_my * df["s37v_m"].mean() + df["s37v_b"].mean()

    print('s19h:')
    print(f' slope mean: {df["s19h_m"].mean():9.5f}', end='')
    print(f'      std: {df["s19h_m"].std():9.5f}')
    print(f' intrc mean: {df["s19h_b"].mean():9.5f}', end='')
    print(f'      std: {df["s19h_b"].std():9.5f}')
    print('')
    print(f'    f17 tp s19h ow: {tp_f17_s19h_ow}')
    print(f'    am2 tp s19h ow: {tp_am2_s19h_ow:.2f}')
    print('')
    print(f'    f17 tp s19h fy: {tp_f17_s19h_fy}')
    print(f'    am2 tp s19h fy: {tp_am2_s19h_fy:.2f}')
    print('')
    print(f'    f17 tp s19h my: {tp_f17_s19h_my}')
    print(f'    am2 tp s19h my: {tp_am2_s19h_my:.2f}')
    print('')
    print('')
    print('s19v:')
    print(f' slope mean: {df["s19v_m"].mean():9.5f}', end='')
    print(f'      std: {df["s19v_m"].std():9.5f}')
    print(f' intrc mean: {df["s19v_b"].mean():9.5f}', end='')
    print(f'      std: {df["s19v_b"].std():9.5f}')
    print('')
    print(f'    f17 tp s19v ow: {tp_f17_s19v_ow}')
    print(f'    am2 tp s19v ow: {tp_am2_s19v_ow:.2f}')
    print('')
    print(f'    f17 tp s19v fy: {tp_f17_s19v_fy}')
    print(f'    am2 tp s19v fy: {tp_am2_s19v_fy:.2f}')
    print('')
    print(f'    f17 tp s19v my: {tp_f17_s19v_my}')
    print(f'    am2 tp s19v my: {tp_am2_s19v_my:.2f}')
    print('')
    print('')
    print('s37v:')
    print(f' slope mean: {df["s37v_m"].mean():9.5f}', end='')
    print(f'      std: {df["s37v_m"].std():9.5f}')
    print(f' intrc mean: {df["s37v_b"].mean():9.5f}', end='')
    print(f'      std: {df["s37v_b"].std():9.5f}')
    print('')
    print(f'    f17 tp s37v ow: {tp_f17_s37v_ow}')
    print(f'    am2 tp s37v ow: {tp_am2_s37v_ow:.2f}')
    print('')
    print(f'    f17 tp s37v fy: {tp_f17_s37v_fy}')
    print(f'    am2 tp s37v fy: {tp_am2_s37v_fy:.2f}')
    print('')
    print(f'    f17 tp s37v my: {tp_f17_s37v_my}')
    print(f'    am2 tp s37v my: {tp_am2_s37v_my:.2f}')
else:
    print(f'Could not determine if nh or sh from file: {ifn}')
    raise SystemExit()
