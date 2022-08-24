"""
make_doy_icemask_raw_psn25.py

Create a valid ice mask from BT data
"""

import datetime as dt
import os

import numpy as np

outdir = './bt_doymasks_raw'
os.makedirs(outdir, exist_ok=True)


def get_sat(d):
    if (d >= dt.date(1995, 5, 10)) and (d <= dt.date(2007, 12, 31)):
        return 'f13'
    elif (d >= dt.date(2008, 1, 1)) and (d <= dt.date(2021, 12, 31)):
        return 'f17'
    else:
        print(f'no BT sat found for: {d}')
        return None


if __name__ == '__main__':
    d0 = dt.date(1995, 5, 10)
    # d1 = dt.date(2017, 12, 31)
    d1 = dt.date(2021, 12, 31)

    doymask_n = np.zeros((366 + 1, 448, 304), dtype=np.uint8)
    doymask_s = np.zeros((366 + 1, 332, 316), dtype=np.uint8)

    d = d0
    while d <= d1:
        sat = get_sat(d)
        ymd = d.strftime('%Y%m%d')
        year = d.strftime('%Y')
        doy = int(d.strftime('%j'))

        if (doy % 30) == 0:
            print(f'  Working on: {d}')

        try:
            nfn = f'/data/nsidc0079/final-gsfc/north/daily/{year}/bt_{ymd}_{sat}_v3.1_n.bin'  # noqa
            ndata = np.fromfile(nfn, dtype=np.int16).reshape(448, 304)
            is_ext_n = (ndata >= 38) & (ndata <= 1000)
            doymask_n[doy][is_ext_n] = 100
        except FileNotFoundError:
            print(f'No file for: {d}')

        try:
            sfn = f'/data/nsidc0079/final-gsfc/south/daily/{year}/bt_{ymd}_{sat}_v3.1_s.bin'  # noqa
            sdata = np.fromfile(sfn, dtype=np.int16).reshape(332, 316)
            is_ext_s = (sdata >= 38) & (sdata <= 1000)
            doymask_s[doy][is_ext_s] = 100
        except FileNotFoundError:
            print(f'No file for: {d}')

        d += dt.timedelta(days=1)

    for doy in range(1, 366 + 1):
        ofnn = f'{outdir}/bt_doymask_n_{doy:03d}.dat'
        doymask_n[doy].tofile(ofnn)

        ofns = f'{outdir}/bt_doymask_s_{doy:03d}.dat'
        doymask_s[doy].tofile(ofns)

        if (doy % 20) == 0:
            print(f'Wrote masks for: {doy}')
