"""
comput_tb_linreg_psn25.py

Compute linear regressions for day(s) with F17 and AMSR2 data
"""

import datetime as dt

# import os
import sys

import numpy as np
from h5py import File
from sklearn.linear_model import LinearRegression

f17_to_au = {
    'n19h': '18H_DAY',
    'n19v': '18V_DAY',
    'n22v': '23V_DAY',
    'n37h': '36H_DAY',
    'n37v': '36V_DAY',
    's19h': '18H_DAY',
    's19v': '18V_DAY',
    's22v': '23V_DAY',
    's37h': '36H_DAY',
    's37v': '36V_DAY',
}


def xwm(m='exiting in xwm()'):
    raise SystemExit(m)


def compute_linreg(
    oldsat,
    newsat,
    d,
    fn_landmask='./psn25_expanded_landmask.dat',
    fn_old_='/data/nsidc0001_f17_2021/tb_f17_{ymd}_v6_{chan}.bin',
    fn_new_='/data/ausi25//AMSR_U2_L3_SeaIce25km_B04_{ymd}.he5',
    fn_valid_='./bt_doymasks_{hem}/bt_doymask_{h}_{doy:03d}.dat',
    hem='nh',
):

    # Compute linear regression for tbs for newsat from oldsat
    if hem == 'nh':
        xdim = 304
        ydim = 448
        h = 'n'
        H = 'N'
        HEM = 'NH'
    elif hem == 'sh':
        xdim = 316
        ydim = 332
        h = 's'
        H = 'S'
        HEM = 'SH'
    else:
        xwm(f'Could not determine dims for hem: {hem}')

    chan_list = (
        f'{h}19h',
        f'{h}19v',
        f'{h}22v',
        f'{h}37h',
        f'{h}37v',
    )

    if (d.month == 1) and (d.day == 1):
        print('   date   ', end='')
        for chan in chan_list:
            print(f'    {chan}_m     {chan}_b  ', end='')
        print(' ')

    print(f'{d}', end='')

    ymd = d.strftime('%Y%m%d')
    doy = int(d.strftime('%j'))

    # landmask is 0 where ocean, 30-32
    landmask = np.fromfile(fn_landmask, dtype=np.uint8).reshape(ydim, xdim)

    # Valid files have value 100 where ice was historically found
    valid = np.fromfile(
        fn_valid_.format(hem=hem, h=h, doy=doy), dtype=np.uint8
    ).reshape(ydim, xdim)

    is_valid_mask = (landmask == 0) & (valid != 0)

    h = hem[0]

    ds_new = File(fn_new_.format(ymd=ymd), 'r')

    for chan in chan_list:
        fn_old = fn_old_.format(ymd=ymd, chan=chan)  # .bin
        tb_old = np.divide(
            np.fromfile(fn_old, dtype=np.int16).reshape(ydim, xdim), 10.0
        )

        tb_new = np.divide(
            np.array(
                ds_new['HDFEOS']['GRIDS'][f'{H}pPolarGrid25km']['Data Fields'][
                    f'SI_25km_{HEM}_{f17_to_au[chan]}'
                ]
            ),  # noqa
            10.0,
        )

        is_valid = is_valid_mask & (tb_old > 50) & (tb_new > 50)

        lr = LinearRegression()
        lr.fit(tb_old[is_valid].reshape(-1, 1), tb_new[is_valid])

        # Verify that we have the direction correct...
        # tb_old_mod = tb_old * lr.coef_ + lr.intercept_
        # lr = LinearRegression()
        # lr.fit(tb_old_mod[is_valid].reshape(-1, 1), tb_new[is_valid])
        # print(f'     Check: LinearRegression for {chan}:')
        # print(f'       slope: {lr.coef_}   intrc: {lr.intercept_}')

        slope = float(lr.coef_)
        intrc = float(lr.intercept_)
        print(f'{slope:10.5f} {intrc:10.5f}  ', end='')

    print('', flush=True)


if __name__ == '__main__':
    try:
        ymd_arg = sys.argv[1]
        ymd = dt.datetime.strptime(ymd_arg, '%Y%m%d').date()
    except IndexError:
        ymd = 'do all'
        # xwm('No ymd given')
    except ValueError:
        xwm(f'Could not get ymd from: {ymd_arg}')

    if ymd == 'do all':
        d = dt.date(2021, 1, 1)
        while d <= dt.date(2021, 12, 31):
            compute_linreg('f17', 'a2', d)

            d += dt.timedelta(days=1)
    else:
        compute_linreg('f17', 'a2', ymd)
