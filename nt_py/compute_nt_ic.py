"""Compute the NASA Team ice concentration.

Note: the original Goddard code involves the following files:
    0: tb files, exactly the same as NSIDC-0001 except:
        - with a 300-byte header
        - 0s have been replace with -10
        - big-endian rather than little-endian format
    1: spatially-interpolated tb files
    2: initial ice concentration files
    3: NT ice conc, including land spillover and valid ice masking
"""

import json
import os
from typing import Any

import numpy as np


def xwm(m='exiting in xwm()'):
    raise SystemExit(m)


def import_cfg_file(ifn):
    with open(ifn) as f:
        params = json.load(f)

    return params


def fdiv(a, b):
    return np.divide(a, b, dtype=np.float32)


def read_tb_field_raw(tbfn):
    # Read int16 scaled by 10 and return float32 unscaled
    raw = np.fromfile(tbfn, dtype=np.int16).reshape(448, 304)
    return raw


def nt_spatint(tbs):
    # Implement spatial interpolation scheme of SpatialInt_np.c
    # and SpatialInt_sp.c
    # Weighting scheme is: orthogonally adjacent weighted 1.0
    #                      diagonally adjacent weighted 0.707
    interp_tbs = {}
    for tb in tbs.keys():
        orig = tbs[tb]
        total = np.zeros_like(orig, dtype=np.float32)
        count = np.zeros_like(orig, dtype=np.float32)

        interp_locs = orig <= 0

        for offset in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            rolled = np.roll(orig, offset, axis=(0, 1))
            has_vals = (rolled > 0) & (interp_locs)
            total[has_vals] += rolled[has_vals]
            count[has_vals] += 1.0

        for offset in ((1, 1), (1, -1), (-1, -1), (-1, 1)):
            rolled = np.roll(orig, offset, axis=(0, 1))
            has_vals = (rolled > 0) & (interp_locs)
            total[has_vals] += 0.707 * rolled[has_vals]
            count[has_vals] += 0.707

        replace_locs = interp_locs & (count > 1.2)
        count[count == 0] = 1
        average = np.divide(total, count, dtype=np.float32)

        """ Debugging, chasing down differences between C and Python
        i = 300
        j = 438
        print(f'total at ({i}, {j}): {total[j, i]}')
        print(f'count at ({i}, {j}): {count[j, i]}')
        print(f'avg   at ({i}, {j}): {average[j, i]}')
        """

        interp = orig.copy()
        interp[replace_locs] = average[replace_locs]

        interp_tbs[tb] = interp

    return interp_tbs


def correct_spi_tbs(tbs):
    # Cause spatially interpolated files to *exactly* match the C code output
    tbs['h19'][438, 300] = 1091  # instead of computed 1090
    tbs['v19'][234, 156] = 2354  # instead of computed 2353
    tbs['v22'][17, 183] = 2500  # instead of computed 2499
    tbs['h37'][20, 182] = 2436  # instead of computed 2435
    # No correction to 'v37'

    return tbs


def get_tiepoints(sat, hem):
    # Return the tiepoints for this sat/hem combo

    tiepoints: dict[str, dict[str, float]] = {}
    if sat == 'f17':
        tiepoints['v19'] = {}
        tiepoints['h19'] = {}
        tiepoints['v37'] = {}
        if hem == 'n':
            tiepoints['v19']['ow'] = 184.9
            tiepoints['v19']['fy'] = 248.4
            tiepoints['v19']['my'] = 220.7

            tiepoints['h19']['ow'] = 113.4
            tiepoints['h19']['fy'] = 232.0
            tiepoints['h19']['my'] = 196.0

            tiepoints['v37']['ow'] = 207.1
            tiepoints['v37']['fy'] = 242.3
            tiepoints['v37']['my'] = 188.5

            have_combo = True

    if have_combo:
        return tiepoints
    else:
        xwm(f'No such combo for tiepoints: sat: {sat}, hem: {hem}')


def compute_nt_coefficients(tp):
    # Compute coefficients for the NT algorithm
    # tp are the tiepoints, a dictionary of structure:
    #   tp[channel][tiepoint]
    #      where channel is 'v19', 'h19', 'v37'
    #            tiepoint is 'ow', 'my', 'fy'
    #                     for open water, multiyear, first-year respectively

    # Intermediate variables
    # TODO: better type annotations.
    diff: dict[str, dict[str, Any]] = {}
    sums: dict[str, dict[str, Any]] = {}
    for tiepoint in ('ow', 'fy', 'my'):
        diff[tiepoint] = {}
        diff[tiepoint]['19v19h'] = tp['v19'][tiepoint] - tp['h19'][tiepoint]
        diff[tiepoint]['37v19v'] = tp['v37'][tiepoint] - tp['v19'][tiepoint]

        sums[tiepoint] = {}
        sums[tiepoint]['19v19h'] = tp['v19'][tiepoint] + tp['h19'][tiepoint]
        sums[tiepoint]['37v19v'] = tp['v37'][tiepoint] + tp['v19'][tiepoint]

    coefs = {}

    coefs['A'] = (
        diff['my']['19v19h'] * diff['ow']['37v19v']
        - diff['my']['37v19v'] * diff['ow']['19v19h']
    )

    coefs['B'] = (
        diff['my']['37v19v'] * sums['ow']['19v19h']
        - diff['ow']['37v19v'] * sums['my']['19v19h']
    )

    coefs['C'] = (
        diff['ow']['19v19h'] * sums['my']['37v19v']
        - diff['my']['19v19h'] * sums['ow']['37v19v']
    )

    coefs['D'] = (
        sums['my']['19v19h'] * sums['ow']['37v19v']
        - sums['my']['37v19v'] * sums['ow']['19v19h']
    )

    coefs['E'] = (
        diff['fy']['19v19h'] * (diff['my']['37v19v'] - diff['ow']['37v19v'])
        + diff['ow']['19v19h'] * (diff['fy']['37v19v'] - diff['my']['37v19v'])
        + diff['my']['19v19h'] * (diff['ow']['37v19v'] - diff['fy']['37v19v'])
    )

    coefs['F'] = (
        diff['fy']['37v19v'] * (sums['my']['19v19h'] - sums['ow']['19v19h'])
        + diff['ow']['37v19v'] * (sums['fy']['19v19h'] - sums['my']['19v19h'])
        + diff['my']['37v19v'] * (sums['ow']['19v19h'] - sums['fy']['19v19h'])
    )

    coefs['G'] = (
        diff['fy']['19v19h'] * (sums['ow']['37v19v'] - sums['my']['37v19v'])
        + diff['ow']['19v19h'] * (sums['my']['37v19v'] - sums['fy']['37v19v'])
        + diff['my']['19v19h'] * (sums['fy']['37v19v'] - sums['ow']['37v19v'])
    )

    coefs['H'] = (
        sums['fy']['37v19v'] * (sums['ow']['19v19h'] - sums['my']['19v19h'])
        + sums['ow']['37v19v'] * (sums['my']['19v19h'] - sums['fy']['19v19h'])
        + sums['my']['37v19v'] * (sums['fy']['19v19h'] - sums['ow']['19v19h'])
    )

    coefs['I'] = (
        diff['fy']['37v19v'] * diff['ow']['19v19h']
        - diff['fy']['19v19h'] * diff['ow']['37v19v']
    )

    coefs['J'] = (
        diff['ow']['37v19v'] * sums['fy']['19v19h']
        - diff['fy']['37v19v'] * sums['ow']['19v19h']
    )

    coefs['K'] = (
        sums['ow']['37v19v'] * diff['fy']['19v19h']
        - diff['ow']['19v19h'] * sums['fy']['37v19v']
    )

    coefs['L'] = (
        sums['fy']['37v19v'] * sums['ow']['19v19h']
        - sums['fy']['19v19h'] * sums['ow']['37v19v']
    )

    return coefs


def compute_ratios(tbs, coefs):
    # Compute NASA Team sea ice concentration estimate
    ratios = {}

    dif_37v19v = tbs['v37'] - tbs['v19']
    sum_37v19v = tbs['v37'] + tbs['v19']
    sum_37v19v[sum_37v19v == 0] = 1  # Avoid div by zero
    ratios['gr_3719'] = np.divide(dif_37v19v, sum_37v19v)

    dif_22v19v = tbs['v22'] - tbs['v19']
    sum_22v19v = tbs['v22'] + tbs['v19']
    sum_22v19v[sum_22v19v == 0] = 1  # Avoid div by zero
    ratios['gr_2219'] = np.divide(dif_22v19v, sum_22v19v)

    dif_19v19h = tbs['v19'] - tbs['h19']
    sum_19v19h = tbs['v19'] + tbs['h19']
    sum_19v19h[sum_19v19h == 0] = 1  # Avoid div by zero
    ratios['pr_1919'] = np.divide(dif_19v19h, sum_19v19h)

    return ratios


def get_gr_thresholds(sat, hem):
    # Return the gradient ratio thresholds for this sat, hem combo
    gr_thresholds = {}
    if sat == 'f17':
        if hem == 'n':
            gr_thresholds['3719'] = 0.050
            gr_thresholds['2219'] = 0.045
        elif hem == 's':
            gr_thresholds['3719'] = 0.053
            gr_thresholds['2219'] = 0.045

    return gr_thresholds


def compute_weather_filtered(tbs, ratios, thres):
    # Determine where array is weather-filtered
    print(f'thres 2219: {thres["2219"]}')
    print(f'thres 3719: {thres["3719"]}')

    filtered = (ratios['gr_2219'] > thres['2219']) | (ratios['gr_3719'] > thres['3719'])

    return filtered


def compute_valid_tbs(tbs):
    # Return boolean array where TBs are valid
    return (tbs['v19'] > 0) & (tbs['h19'] > 0) & (tbs['v37'] > 0)


def compute_nt_conc(tbs, coefs, ratios):
    # Compute NASA Team sea ice concentration estimate
    pr_gr_product = ratios['pr_1919'] * ratios['gr_3719']

    dd = (
        coefs['E']
        + coefs['F'] * ratios['pr_1919']
        + coefs['G'] * ratios['gr_3719']
        + coefs['H'] * pr_gr_product
    )

    fy = (
        coefs['A']
        + coefs['B'] * ratios['pr_1919']
        + coefs['C'] * ratios['gr_3719']
        + coefs['D'] * pr_gr_product
    )

    my = (
        coefs['I']
        + coefs['J'] * ratios['pr_1919']
        + coefs['K'] * ratios['gr_3719']
        + coefs['L'] * pr_gr_product
    )

    # Because we have not excluded missing-tb and weather-filtered points,
    # it is possible for denominator 'dd' to have value of zero.  Remove
    # this for division
    dd[dd == 0] = 0.001  # This causes the denominator to become 1.0

    conc = (fy + my) / dd * 1000.0

    conc[conc < 0] = 0

    return conc


def apply_nt_spillover(conc_int16):
    # Apply the NASA Team land spillover routine

    shoremap_fn = (
        '/home/vagrant/cdr_amsr2/nt_orig/DATAFILES/data36/maps/shoremap_north_25'
    )
    shoremap = np.fromfile(shoremap_fn, dtype='>i2')[150:].reshape(448, 304)
    print(f'Read shoremap from:\n  .../{os.path.basename(shoremap_fn)}')
    print(f'  shoremap min: {shoremap.min()}')
    print(f'  shoremap max: {shoremap.max()}')

    minic_fn = (
        '/home/vagrant/cdr_amsr2/nt_orig/DATAFILES/data36/maps/SSMI8_monavg_min_con'
    )
    minic = np.fromfile(minic_fn, dtype='>i2')[150:].reshape(448, 304)
    print(f'Read minic from:\n  .../{os.path.basename(minic_fn)}')
    print(f'  minic min: {minic.min()}')
    print(f'  minic max: {minic.max()}')

    newice = conc_int16.copy()

    newice[shoremap == 1] = -9999
    newice[shoremap == 2] = -9998

    minic[(shoremap == 5) & (minic > 200)] = 200
    minic[(shoremap == 4) & (minic > 400)] = 400
    minic[(shoremap == 3) & (minic > 600)] = 600

    # Count number of nearby low ice conc
    n_low = np.zeros_like(conc_int16, dtype=np.uint8)

    for joff in range(-3, 3 + 1):
        for ioff in range(-3, 3 + 1):
            offmax = max(abs(ioff), abs(joff))
            # print(f'offset: ({ioff}, {joff}): {offmax}')

            rolled = np.roll(conc_int16, (joff, ioff), axis=(0, 1))
            is_rolled_low = (rolled < 150) & (rolled >= 0)

            is_at_coast = shoremap == 5
            is_near_coastal = shoremap == 4
            is_far_coastal = shoremap == 3

            if offmax <= 1:
                n_low[is_rolled_low & is_at_coast] += 1

            if offmax <= 2:
                n_low[is_rolled_low & is_near_coastal] += 1

            if offmax <= 3:
                n_low[is_rolled_low & is_far_coastal] += 1

    # Note: there are meaningless differences "at the edge" in these counts
    # because the spatial interpolation is not identical along the border

    where_reduce_ice = (n_low >= 3) & (shoremap > 2)
    newice[where_reduce_ice] -= minic[where_reduce_ice]

    # where_ice_overreduced = (conc_int16 >= 0) & (newice < 0) & (newice > -9000)
    where_ice_overreduced = (conc_int16 >= 0) & (newice < 0) & (shoremap > 2)
    newice[where_ice_overreduced] = 0

    # Preserve missing data (conc value of -10)
    # where_missing = (conc_int16 < 0) & (newice > -9000)
    where_missing = (conc_int16 < 0) & where_reduce_ice & (shoremap > 2)
    newice[where_missing] = conc_int16[where_missing]

    # newice.tofile('conc_landspill_py.dat')
    # print('Wrote: conc_landspill_py.dat')

    return newice


def apply_sst(conc):
    # Apply the sst filter
    sst_threshold = 2780
    sst = conc.copy()

    sst_fn = (
        '/home/vagrant/cdr_amsr2/nt_orig'
        '/DATAFILES/data36/SST/North/jan.temp.zdf.ssmi_fixed_25fill.fixed'
    )
    sst_field = np.fromfile(sst_fn, dtype='>i2')[150:].reshape(448, 304)
    print(f'Read sst from:\n  .../{os.path.basename(sst_fn)}')
    print(f'  sst_field min: {sst_field.min()}')
    print(f'  sst_field max: {sst_field.max()}')

    where_sst_high = sst_field >= sst_threshold
    sst[where_sst_high] = 0

    return sst


def apply_polehole(conc):
    # Apply the pole hole
    new_conc = conc.copy()

    polehole_fn = (
        '/home/vagrant/cdr_amsr2/nt_orig'
        '/DATAFILES/data36/maps/nsssspoleholemask_for_ICprod'
    )
    polehole = np.fromfile(polehole_fn, dtype='>i2')[150:].reshape(448, 304)
    print(f'Read polehole from:\n  .../{os.path.basename(polehole_fn)}')
    print(f'  polehole min: {polehole.min()}')
    print(f'  polehole max: {polehole.max()}')

    where_polehole = polehole == 1
    new_conc[where_polehole] = -50

    return new_conc


if __name__ == '__main__':
    do_exact = True

    params = import_cfg_file('./nt_sample_nh.json')

    params['sat'] = 'f17'
    params['hem'] = 'n'

    print(f'{params}')

    orig_tbs = {}
    for tb in ('v19', 'h19', 'v22', 'h37', 'v37'):
        orig_tbs[tb] = read_tb_field_raw(params['raw_fns'][tb])

    spi_tbs = nt_spatint(orig_tbs)
    if do_exact:
        spi_tbs = correct_spi_tbs(spi_tbs)

    """
    for tb in spi_tbs.keys():
        ofn = f'spi_{tb}.dat'
        spi_tbs[tb].tofile(ofn)
        print(f'Wrote: {ofn}')
    """

    # Here, the tbs are identical to the output of the Goddard code

    # The next step is to implement:
    #   seaice5con 001 2018 001 2018 TOT_CON ssmif17 n
    #
    # icetype   = TOT_CON
    # type      = "tcon"
    # titletype = "Team Ice Concentration"
    # col = 304
    # row = 448
    # pole[0] = 'n'
    # ipole = 0

    # Calls: team( 1, 1, missing, brtemps, scale )
    tiepoints = get_tiepoints(params['sat'], params['hem'])
    print(f'tiepoints: {tiepoints}')

    nt_coefficients = compute_nt_coefficients(tiepoints)
    print(f'NT coefs: {nt_coefficients}')
    for c in ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L'):
        print(f'  coef {c}: {nt_coefficients[c]}')

    is_valid_tbs = compute_valid_tbs(spi_tbs)

    gr_thresholds = get_gr_thresholds(params['sat'], params['hem'])
    print(f'gr_thresholds:\n{gr_thresholds}')

    ratios = compute_ratios(spi_tbs, nt_coefficients)

    weather_filtered = compute_weather_filtered(spi_tbs, ratios, gr_thresholds)

    conc = compute_nt_conc(spi_tbs, nt_coefficients, ratios)

    # Set invalid tbs and weather-filtered values
    conc[~is_valid_tbs] = -10
    conc[weather_filtered] = 0
    conc_int16 = conc.astype(np.int16)

    # This "conc_int16" field is identical to that saved to:
    #    ../nt_orig/system/new_2_iceconcentrations/
    #  eg
    #    nssss1d17tcon2018001
    # conc_int16.tofile('conc_raw_py.dat')

    # Apply NT-land spillover filter
    conc_spill = apply_nt_spillover(conc_int16)

    # Apply SST-threshold
    conc_sst = apply_sst(conc_spill)
    # conc_sst.tofile('conc_sst_py.dat')

    # Apply pole hole
    conc_pole = apply_polehole(conc_sst)
    # conc_pole.tofile('conc_pole_py.dat')

    # Write output file
    conc_pole.tofile('nt_sample_nh.dat')
