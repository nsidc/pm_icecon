"""Compute the Bootstrap ice concentration.

Takes values from part of boot_ice_sb2_ssmi_np_nrt.f
and computes:
    iceout
"""

import json
from functools import reduce
from pathlib import Path
from typing import Sequence

import numpy as np
import numpy.typing as npt

from bt_py._types import Params, ParaVals, Variables
from bt_py.errors import BootstrapAlgError, UnexpectedSatelliteError

THIS_DIR = Path(__file__).parent


def f(num):
    # return float32 of num
    return np.float32(num)


def import_cfg_file(ifn: Path):
    return json.loads(ifn.read_text())


def read_tb_field(tbfn: Path) -> npt.NDArray[np.float32]:
    # Read int16 scaled by 10 and return float32 unscaled
    raw = np.fromfile(tbfn, dtype=np.int16).reshape(448, 304)

    return fdiv(raw.astype(np.float32), 10)


def tb_data_mask(
    *,
    tbs: Sequence[npt.NDArray[np.float32]],
    min_tb: float,
    max_tb: float,
) -> npt.NDArray[np.bool_]:
    """Return a boolean ndarray inidcating areas of bad data.

    Bad data are locations where any of the given Tbs are outside the range
    defined by (mintb, maxtb)

    True values indicate bad data that should be masked. False values indicate
    good data.
    """

    def _is_outofrange_tb(tb, min_tb, max_tb):
        return (tb < min_tb) | (tb > max_tb)

    is_bad_tb = reduce(
        np.logical_or,
        [_is_outofrange_tb(tb, min_tb, max_tb) for tb in tbs],
    )

    return is_bad_tb


def xfer_tbs_nrt(v37, h37, v19, v22, sat) -> dict[str, npt.NDArray[np.float32]]:
    # NRT regressions
    if sat == '17':
        v37 = fadd(fmul(1.0170066, v37), -4.9383355)
        h37 = fadd(fmul(1.0009720, h37), -1.3709822)
        v19 = fadd(fmul(1.0140723, v19), -3.4705583)
        v22 = fadd(fmul(0.99652931, v22), -0.82305684)
    elif sat == '18':
        v37 = fadd(fmul(1.0104497, v37), -3.3174017)
        h37 = fadd(fmul(0.98914390, h37), 1.2031835)
        v19 = fadd(fmul(1.0057373, v19), -0.92638520)
        v22 = fadd(fmul(0.98793409, v22), 1.2108198)
    else:
        raise UnexpectedSatelliteError(f'No such sat tb xform: {sat}')

    return {
        'v37': v37,
        'h37': h37,
        'v19': v19,
        'v22': v22,
    }


def ret_adj_adoff(wtp, vh37, perc=0.92):
    # replaces ret_adj_adoff()
    # wtp is two water tie points
    # vh37 is offset and slope
    wtp1, wtp2 = f(wtp[0]), f(wtp[1])
    off, slp = f(vh37[0]), f(vh37[1])

    x = ((wtp1 / slp) + wtp2 - off) / (slp + 1.0 / slp)
    y = slp * x + off

    dx = wtp1 - x
    dx2 = perc * dx
    x2 = wtp1 - dx2

    dy = y - wtp2
    dy2 = perc * dy
    y2 = wtp2 + dy2

    new_off = y2 - slp * x2

    adoff = off - new_off

    return adoff


def ret_para_nsb2(tbset, sat, season) -> ParaVals:
    # reproduce effect of ret_para_nsb2()
    # Note: instead of '1' or '2', use description of axes tb1 and tb2
    #       to identify the TB set whose parameters are being set
    #       So, tbset is 'v1937' or 'vh37'
    # Note: 'sat' is a *string*, not an integer

    # Set wintrc, wslope, wxlimt
    if sat == '00':
        if season == 1:
            wintrc = 53.4153
            wslope = 0.661017
            wxlimt = 22.00
        else:
            wintrc = 60.1667
            wslope = 0.633333
            wxlimt = 24.00
    else:
        if season == 1:
            if sat != '17' and sat != '18':
                wintrc = 90.3355
                wslope = 0.501537
            else:
                wintrc = 87.6467
                wslope = 0.517333
            wxlimt = 14.00
        else:
            if sat != '17' and sat != '18':
                wintrc = 89.3316
                wslope = 0.501537
            else:
                wintrc = 89.2000
                wslope = 0.503750
            wxlimt = 21.00

    if tbset == 'vh37':
        wtp = [201.916, 132.815]
        itp = [255.670, 241.713]
        lnline = [-73.5471, 1.21104]
        iceline = [-25.9729, 1.04382]
        lnchk = 1.5
    elif tbset == 'v1937':
        wtp = [201.916, 178.771]
        itp = [255.670, 258.341]
        lnline = [47.0061, 0.809335]
        iceline = [112.803, 0.550296]
        lnchk = 1.5

    return {
        'wintrc': wintrc,
        'wslope': wslope,
        'wxlimt': wxlimt,
        'wtp': wtp,
        'itp': itp,
        'lnline': lnline,
        'iceline': iceline,
        'lnchk': lnchk,
    }


def ret_wtp_32(water_arr: npt.NDArray[np.int16], tb: npt.NDArray[np.float32]) -> float:
    # Attempt to reproduce Goddard methodology for computing water tie point
    # v['wtp19v'] = ret_wtp_32(water_arr, v19, wtp19v)

    # Note: this *really* should be done with np.percentile()

    pct = 0.02

    # Compute quarter-Kelvin histograms
    histo, histo_edges = np.histogram(
        tb[water_arr == 1],
        bins=1200,
        range=(0, 300),
    )
    nvals = histo.sum()

    # Remove low-count bins (but don't adjust total ??!!)
    histo[histo <= 10] = 0

    ival = 0
    subtotal = 0
    thresh = f(nvals) * pct

    while (ival < 1200) and (subtotal < thresh):
        subtotal += histo[ival]
        ival += 1

    ival -= 1  # undo last increment

    # TODO: this expression returns `np.float64`, NOT `np.float32` like `f`
    # returns...
    wtp = f(ival) * 0.25

    return wtp


def linfit_32(xvals, yvals):
    # Implement original Bootstrap linear-fit routine
    nvals = f(xvals.shape[0])
    sumx = np.sum(xvals, dtype=np.float64)
    sumy = np.sum(yvals, dtype=np.float64)
    sumx2 = np.sum(fsqr(xvals), dtype=np.float64)
    # sumy2 is included in Bootstrap, but not used.
    # sumy2 = np.sum(fsqr(yvals), dtype=np.float64)
    sumxy = np.sum(fmul(xvals, yvals), dtype=np.float64)

    # float32 version
    # delta = fsub(fmul(nvals, sumx2), fsqr(sumx))
    # offset = fdiv(fsub(fmul(sumx2, sumy), fmul(sumx, sumxy)), delta)
    # slope = fdiv(fsub(fmul(sumxy, nvals), fmul(sumx, sumy)), delta)

    # float64 version
    delta = (nvals * sumx2) - sumx * sumx
    offset = ((sumx2 * sumy) - (sumx * sumxy)) / delta
    slope = ((sumxy * nvals) - (sumx * sumy)) / delta

    return offset, slope


def ret_linfit_32(
    land,
    tb_mask: npt.NDArray[np.bool_],
    tbx,
    tby,
    lnline,
    lnchk,
    add,
    water,
    tba=None,
    iceline=None,
    adoff=None,
):
    # Reproduces both ret_linfit1() and ret_linfit2()
    # Note: lnline is two, 0 is offset, 1 is slope
    # Note: iceline is two, 0 is offset, 1 is slope

    not_land_or_masked = (land == 0) & ~tb_mask
    if tba is not None:
        is_tba_le_modad = tba <= fadd(fmul(tbx, iceline[1]), fsub(iceline[0], adoff))
    else:
        is_tba_le_modad = np.full_like(not_land_or_masked, fill_value=True)

    is_tby_gt_lnline = tby > fadd(fmul(tbx, lnline[1]), lnline[0])

    is_water0 = water == 0

    is_valid = not_land_or_masked & is_tba_le_modad & is_tby_gt_lnline & is_water0

    icnt = np.sum(np.where(is_valid, 1, 0))

    xvals = tbx[is_valid].astype(np.float32).flatten().astype(np.float64)
    yvals = tby[is_valid].astype(np.float32).flatten().astype(np.float64)

    if icnt > 125:
        intrca, slopeb = linfit_32(xvals, yvals)
        fit_off = fadd(intrca, add)
        fit_slp = f(slopeb)
    else:
        raise BootstrapAlgError(f'Insufficient valid linfit points: {icnt}')

    return [fit_off, fit_slp]


def ret_ic_32(tbx, tby, wtpx, wtpy, iline_off, iline_slp, baddata, maxic):

    delta_x = tbx - wtpx
    is_deltax_eq_0 = delta_x == 0

    # block1
    y_intercept = iline_off + iline_slp * tbx
    length1 = tby - wtpy
    length2 = y_intercept - wtpy
    ic_block1 = length1 / length2
    ic_block1[ic_block1 < 0] = 0
    ic_block1[ic_block1 > maxic] = maxic

    # block2
    delta_y = tby - wtpy
    slope = delta_y / delta_x
    offset = tby - (slope * tbx)
    slp_diff = iline_slp - slope

    is_slp_diff_ne_0 = slp_diff != 0

    x_intercept = (offset - iline_off) / slp_diff
    y_intercept = offset + (slope * x_intercept)
    length1 = np.sqrt(np.square(tbx - wtpx) + np.square(tby - wtpy))
    length2 = np.sqrt(np.square(x_intercept - wtpx) + np.square(y_intercept - wtpy))
    ic_block2 = length1 / length2
    ic_block2[ic_block2 < 0] = 0
    ic_block2[ic_block2 > maxic] = maxic
    ic_block2[~is_slp_diff_ne_0] = baddata

    # Assume ic is block2, then overwrite if block1
    ic = ic_block2
    ic[is_deltax_eq_0] = ic_block1[is_deltax_eq_0]

    return ic


def fadd(a: npt.ArrayLike, b: npt.ArrayLike) -> npt.NDArray[np.float32]:
    return np.array(np.add(a, b, dtype=np.float32))


def fsub(a: npt.ArrayLike, b: npt.ArrayLike) -> npt.NDArray[np.float32]:
    return np.array(np.subtract(a, b, dtype=np.float32))


def fmul(a: npt.ArrayLike, b: npt.ArrayLike) -> npt.NDArray[np.float32]:
    return np.array(np.multiply(a, b, dtype=np.float32))


def fdiv(a: npt.ArrayLike, b: npt.ArrayLike) -> npt.NDArray[np.float32]:
    return np.array(np.divide(a, b, dtype=np.float32))


def fsqr(a: npt.ArrayLike) -> npt.NDArray[np.float32]:
    return np.array(np.square(a, dtype=np.float32))


def fsqt(a: npt.ArrayLike) -> npt.NDArray[np.float32]:
    return np.array(np.sqrt(a, dtype=np.float32))


def ret_water_ssmi(
    v37,
    h37,
    v22,
    v19,
    land,
    tb_mask: npt.NDArray[np.bool_],
    wslope,
    wintrc,
    wxlimt,
    ln1,
) -> npt.NDArray[np.int16]:
    # Determine where there is definitely water
    not_land_or_masked = (land == 0) & ~tb_mask
    watchk1 = fadd(fmul(f(wslope), v22), f(wintrc))
    watchk2 = fsub(v22, v19)
    watchk4 = fadd(fmul(ln1[1], v37), ln1[0])

    is_cond1 = (watchk1 > v19) | (watchk2 > wxlimt)
    is_cond2 = (watchk4 > h37) | (v37 >= 230.0)

    is_water = not_land_or_masked & is_cond1 & is_cond2

    water = np.zeros_like(land, dtype=np.int16)
    water[is_water] = 1

    return water


def calc_rad_coeffs_32(p, v):
    # Compute radlsp, radoff, radlen vars
    # from (p)arameters and (v)ariables

    v['radslp1'] = fdiv(
        fsub(f(v['itp'][1]), f(v['wtp'][1])), fsub(f(v['itp'][0]), f(v['wtp'][0]))
    )
    v['radoff1'] = fsub(f(v['wtp'][1]), fmul(f(v['wtp'][0]), f(v['radslp1'])))
    xint = fdiv(
        fsub(f(v['radoff1']), f(v['vh37'][0])), fsub(f(v['vh37'][1]), f(v['radslp1']))
    )
    yint = fadd(fmul(v['vh37'][1], f(xint)), f(v['vh37'][0]))
    v['radlen1'] = fsqt(
        fadd(fsqr(fsub(f(xint), f(v['wtp'][0]))), fsqr(fsub(f(yint), f(v['wtp'][1]))))
    )

    v['radslp2'] = fdiv(
        fsub(f(v['itp2'][1]), f(v['wtp2'][1])), fsub(f(v['itp2'][0]), f(v['wtp2'][0]))
    )
    v['radoff2'] = fsub(f(v['wtp2'][1]), fmul(f(v['wtp2'][0]), f(v['radslp2'])))
    xint = fdiv(
        fsub(f(v['radoff2']), f(v['v1937'][0])), fsub(f(v['v1937'][1]), f(v['radslp2']))
    )
    yint = fadd(fmul(f(v['v1937'][1]), f(xint)), f(v['v1937'][0]))
    v['radlen2'] = fsqt(
        fadd(fsqr(fsub(f(xint), f(v['wtp2'][0]))), fsqr(fsub(f(yint), f(v['wtp2'][1]))))
    )

    return v


def calc_rad_coeffs(p, v):
    # Compute radlsp, radoff, radlen vars
    # from (p)arameters and (v)ariables

    v['radslp1'] = (v['itp'][1] - v['wtp'][1]) / (v['itp'][0] - v['wtp'][0])
    v['radoff1'] = v['wtp'][1] - v['wtp'][0] * v['radslp1']
    xint = (v['radoff1'] - v['vh37'][0]) / (v['vh37'][1] - v['radslp1'])
    yint = v['vh37'][1] * xint + v['vh37'][0]
    v['radlen1'] = np.sqrt(
        np.square(xint - v['wtp'][0]) + np.square(yint - v['wtp'][1])
    )

    # next...do the radslp2 etc calcs
    v['radslp2'] = (v['itp2'][1] - v['wtp2'][1]) / (v['itp2'][0] - v['wtp2'][0])
    v['radoff2'] = v['wtp2'][1] - v['wtp2'][0] * v['radslp2']
    xint = (v['radoff2'] - v['v1937'][0]) / (v['v1937'][1] - v['radslp2'])
    yint = v['v1937'][1] * xint + v['v1937'][0]
    v['radlen2'] = np.sqrt(
        np.square(xint - v['wtp2'][0]) + np.square(yint - v['wtp2'][1])
    )

    v['radslp1'] = np.float64(v['radslp1'])
    v['radoff1'] = np.float64(v['radoff1'])
    v['radlen1'] = np.float64(v['radlen1'])
    v['radslp2'] = np.float64(v['radslp2'])
    v['radoff2'] = np.float64(v['radoff2'])
    v['radlen2'] = np.float64(v['radlen2'])

    return v


def sst_clean_sb2(iceout, missval, landval, month, pole):
    # implement fortran's sst_clean_sb2() routine
    imonth = int(month)
    sst_fn = (
        THIS_DIR
        / '..'
        / f'SB2_NRT_programs/ANCILLARY/np_sect_sst1_sst2_mask_{imonth:02d}.int'
    ).resolve()
    sst_mask = np.fromfile(sst_fn, dtype=np.int16).reshape(448, 304)

    is_not_land = iceout != landval
    is_not_miss = iceout != missval
    is_high_sst = sst_mask == 24
    is_not_land_miss_sst = is_not_land & is_not_miss & is_high_sst

    ice_sst = iceout.copy()
    ice_sst[is_not_land_miss_sst] = 0.0

    return ice_sst


def spatial_interp(iceout, missval: float, landval: float, nphole_fn: Path):
    # implement fortran's spatial_interp() routine
    if iceout.shape[1] == 304:
        holemask = np.fromfile(nphole_fn, dtype=np.int16).reshape(448, 304)

    # Use -200 as a not-valid ocean sentinel value
    # so that it works with np.roll
    oceanvals = iceout.copy()

    total = np.zeros_like(oceanvals, dtype=np.float32)
    count = np.zeros_like(oceanvals, dtype=np.int32)
    for joff in range(-1, 2):
        for ioff in range(-1, 2):
            rolled = np.roll(oceanvals, (joff, ioff), axis=(1, 0))
            not_land_nor_miss = (rolled != landval) & (rolled != missval)
            total[not_land_nor_miss] += rolled[not_land_nor_miss]
            count[not_land_nor_miss] += 1

    count[count == 0] = 1
    replace_vals = fdiv(total, count)
    replace_locs = (oceanvals == missval) & (count >= 1) & (holemask == 0)

    iceout[replace_locs] = replace_vals[replace_locs]

    return iceout


def coastal_fix(arr, missval, landval, minic):
    # Apply coastal_fix() routine per Bootstrap

    # Calculate 'temp' array
    #   -1 is no ice
    #    1 is safe from removal
    #    0 is might-be-removed
    temp = np.ones_like(arr, dtype=np.int16)
    is_land_or_lowice = (arr == landval) | ((arr >= 0) & (arr < minic))
    temp[is_land_or_lowice] = -1

    is_seaice = (arr > 0) & (arr <= 100.0)

    off_set = (np.array((0, 1)), np.array((0, -1)), np.array((1, 0)), np.array((-1, 0)))

    for offp1 in off_set:
        offn1 = -1 * offp1  # offp1 * -1
        offn2 = -2 * offp1  # offp1 * -2

        # Compute shifted grids
        rolled_offn1 = np.roll(arr, offp1, axis=(1, 0))  # land
        rolled_off00 = arr  # .  k1p0 k2p0
        rolled_offp1 = np.roll(arr, offn1, axis=(1, 0))  # k1 k2p1
        rolled_offp2 = np.roll(arr, offn2, axis=(1, 0))  # k2

        # is_rolled_land = rolled_offn1 == landval
        is_rolled_land = rolled_offn1 == landval

        is_k1 = (
            (is_seaice)
            & (is_rolled_land)
            & (rolled_offp1 >= 0)
            & (rolled_offp1 < minic)
        )
        is_k2 = (
            (is_seaice)
            & (is_rolled_land)
            & (rolled_offp2 >= 0)
            & (rolled_offp2 < minic)
        )

        is_k1p0 = (
            (is_k1)
            & (rolled_off00 > 0)
            & (rolled_off00 != missval)
            & (rolled_off00 != landval)
        )
        is_k2p0 = (
            (is_k2)
            & (rolled_off00 > 0)
            & (rolled_off00 != missval)
            & (rolled_off00 != landval)
        )
        is_k2p1 = (
            (is_k2)
            & (rolled_offp1 > 0)
            & (rolled_offp1 != missval)
            & (rolled_offp1 != landval)
        )

        temp[is_k1p0] = 0
        temp[is_k2p0] = 0

        # Note, the change_locs are offset by the vals in offp1
        #  for p==1
        where_k2p1 = np.where(is_k2p1)
        change_locs_k2p1 = tuple([where_k2p1[0] + offp1[1], where_k2p1[1] + offp1[0]])
        # temp[tuple(change_locs_k2p1)] = 0
        temp[change_locs_k2p1] = 0

    # HERE: temp array has been set

    # Calculate 'arr2' array
    # This is initially a copy of the arr array, but then has values
    #   set to zero where 'appropriate' based on the temp array
    arr2 = arr.copy()

    # This is very complicated to figure out as modification
    # of the series of off_sets.  Simply coding each of the
    # four change sections manually

    # Note: some of these conditional arrays might be set more than 1x

    # Compute shifted arr grid, for land check
    land_check = np.roll(arr, (0, 1), axis=(1, 0))  # land check
    is_rolled_land = land_check == landval

    # For offp1 of [0, 1], the rolls are:
    tip1jp1 = np.roll(temp, (-1, -1), axis=(1, 0))
    tim1jp1 = np.roll(temp, (1, -1), axis=(1, 0))
    tip1jp0 = np.roll(temp, (-1, 0), axis=(1, 0))
    tim1jp0 = np.roll(temp, (1, 0), axis=(1, 0))

    tip0jp1 = np.roll(temp, (0, -1), axis=(1, 0))

    is_temp0 = temp == 0
    is_considered = is_temp0 & is_rolled_land

    is_tip1jp1_lt0 = tip1jp1 <= 0
    is_tim1jp1_lt0 = tim1jp1 <= 0
    is_tip1jp0_lt0 = tip1jp0 <= 0
    is_tim1jp0_lt0 = tim1jp0 <= 0

    is_tip0jp1_eq0 = tip0jp1 == 0

    # Changing arr2(i,j+1) to 0
    locs_ip0jp1 = np.where(
        is_considered & is_tip1jp1_lt0 & is_tim1jp1_lt0 & is_tip0jp1_eq0
    )
    change_locs_arr2_ip0jp1 = tuple([locs_ip0jp1[0] + 1, locs_ip0jp1[1] + 0])
    arr2[change_locs_arr2_ip0jp1] = 0

    # Changing arr2(i,j) to 0
    locs_ip0jp0 = np.where(
        is_considered
        & is_tip1jp1_lt0
        & is_tim1jp1_lt0
        & is_tip1jp0_lt0
        & is_tim1jp0_lt0
    )
    change_locs_arr2_ip0jp0 = tuple([locs_ip0jp0[0], locs_ip0jp0[1]])
    arr2[change_locs_arr2_ip0jp0] = 0

    # Second arr2 change section

    # Compute shifted arr grid, for land check
    land_check = np.roll(arr, (0, -1), axis=(1, 0))  # land check
    is_rolled_land = land_check == landval

    is_temp0 = temp == 0
    is_considered = is_temp0 & is_rolled_land

    # For offp1 of [0, 1], the rolls are:
    # args to np.roll are opposite of fortran index offsets
    tip1jm1 = np.roll(temp, (-1, 1), axis=(1, 0))
    tim1jm1 = np.roll(temp, (1, 1), axis=(1, 0))
    tip0jm1 = np.roll(temp, (0, 1), axis=(1, 0))
    tip1jp0 = np.roll(temp, (-1, 0), axis=(1, 0))
    tim1jp0 = np.roll(temp, (1, 0), axis=(1, 0))

    is_tip1jm1_le0 = tip1jm1 <= 0
    is_tim1jm1_le0 = tim1jm1 <= 0
    is_tip0jm1_eq0 = tip0jm1 == 0
    is_tip1jp0_le0 = tip1jp0 <= 0
    is_tim1jp0_le0 = tim1jp0 <= 0

    # Changing arr2(i,j-1) to 0
    locs_ip0jm1 = np.where(
        is_considered & is_tip1jm1_le0 & is_tim1jm1_le0 & is_tip0jm1_eq0
    )
    change_locs_arr2_ip0jm1 = tuple([locs_ip0jm1[0] - 1, locs_ip0jm1[1] + 0])
    arr2[change_locs_arr2_ip0jm1] = 0

    # Changing arr2(i,j) to 0
    locs_ip0jp0 = np.where(
        is_considered
        & is_tip1jm1_le0
        & is_tim1jm1_le0
        & is_tip1jp0_le0
        & is_tim1jp0_le0
    )
    change_locs_arr2_ip0jp0 = tuple([locs_ip0jp0[0], locs_ip0jp0[1]])
    arr2[change_locs_arr2_ip0jp0] = 0

    # Third arr2 change section

    # Compute shifted arr grid, for land check
    land_check = np.roll(arr, (1, 0), axis=(1, 0))
    is_rolled_land = land_check == landval

    is_temp0 = temp == 0
    is_considered = is_temp0 & is_rolled_land

    # args to np.roll are opposite of fortran index offsets
    tip1jp1 = np.roll(temp, (-1, -1), axis=(1, 0))
    tip1jp0 = np.roll(temp, (-1, 0), axis=(1, 0))
    tip0jm1 = np.roll(temp, (0, 1), axis=(1, 0))
    tip0jp1 = np.roll(temp, (0, -1), axis=(1, 0))

    is_tip1jp1_le0 = tip1jp1 <= 0
    is_tip1jp0_eq0 = tip1jp0 == 0
    is_tip0jm1_le0 = tip0jm1 <= 0
    is_tip0jp1_le0 = tip0jp1 <= 0

    # Changing arr2(i+1,j) to 0
    locs_ip1jp0 = np.where(
        is_considered & is_tip1jp1_le0 & is_tip1jp1_le0 & is_tip1jp0_eq0
    )
    change_locs_arr2_ip1jp0 = tuple([locs_ip1jp0[0] + 0, locs_ip1jp0[1] + 1])
    arr2[change_locs_arr2_ip1jp0] = 0

    # Changing arr2(i,j) to 0
    locs_ip0jp0 = np.where(
        is_considered
        & is_tip1jp1_le0
        & is_tip1jp1_le0
        & is_tip0jm1_le0
        & is_tip0jp1_le0
    )
    change_locs_arr2_ip0jp0 = tuple([locs_ip0jp0[0], locs_ip0jp0[1]])
    arr2[change_locs_arr2_ip0jp0] = 0

    # Fourth section

    # Compute shifted arr grid, for land check
    land_check = np.roll(arr, (-1, 0), axis=(1, 0))
    is_rolled_land = land_check == landval

    is_temp0 = temp == 0
    is_considered = is_temp0 & is_rolled_land

    # args to np.roll are opposite of fortran index offsets
    tim1jm1 = np.roll(temp, (1, 1), axis=(1, 0))
    tim1jp1 = np.roll(temp, (1, -1), axis=(1, 0))
    tim1jp0 = np.roll(temp, (1, 0), axis=(1, 0))
    tip0jm1 = np.roll(temp, (0, 1), axis=(1, 0))
    tip0jp1 = np.roll(temp, (0, -1), axis=(1, 0))

    is_tim1jm1_le0 = tim1jm1 <= 0
    is_tim1jp1_le0 = tim1jp1 <= 0
    is_tim1jp0_eq0 = tim1jp0 == 0
    is_tip0jm1_le0 = tip0jm1 <= 0
    is_tip0jp1_le0 = tip0jp1 <= 0

    # Changing arr2(i-1,j) to 0
    locs_im1jp0 = np.where(
        is_considered & is_tim1jm1_le0 & is_tim1jp1_le0 & is_tim1jp0_eq0
    )
    change_locs_arr2_im1jp0 = tuple([locs_im1jp0[0] + 0, locs_im1jp0[1] - 1])
    arr2[change_locs_arr2_im1jp0] = 0

    # Changing arr2(i,j) to 0
    locs_ip0jp0 = np.where(
        is_considered
        & is_tim1jm1_le0
        & is_tim1jp1_le0
        & is_tip0jm1_le0
        & is_tip0jp1_le0
    )
    change_locs_arr2_ip0jp0 = tuple([locs_ip0jp0[0], locs_ip0jp0[1]])
    arr2[change_locs_arr2_ip0jp0] = 0

    return arr2


def fix_output_gdprod(conc, minval, maxval, landval, missval):
    # Return fixout, a 2-byte integer field with ice concentration
    fixout = np.zeros_like(conc, dtype=np.int16)
    scaling_factor = 10.0

    is_seaice = (conc >= minval) & (conc <= maxval)
    is_pos_seaice = is_seaice & (conc > 0)
    fixout[is_pos_seaice] = fadd(fmul(conc[is_pos_seaice], scaling_factor), 0.5).astype(
        np.int16
    )

    is_neg_seaice = is_seaice & (conc <= 0)
    fixout[is_neg_seaice] = fsub(fmul(conc[is_neg_seaice], scaling_factor), 0.5).astype(
        np.int16
    )

    is_land = conc == landval
    fixout[is_land] = fmul(conc[is_land], scaling_factor).astype(np.int16)

    is_missing = conc == missval
    fixout[is_missing] = fmul(conc[is_missing], scaling_factor).astype(np.int16)

    return fixout


def get_satymd_from_tbfn(fn):
    # expect filename of format:
    #  ../SB2_NRT_programs/orig_input_tbs/tb_f18_20180217_nrt_n37v.bin
    # Note: this is *extremely* hard-coded
    import os

    bfn = os.path.basename(fn)
    print(f'bfn: {bfn}')
    sat = bfn[4:6]
    year = bfn[7:11]
    month = bfn[11:13]
    day = bfn[13:15]

    return sat, year, month, day


def calc_bt_ice(
    p,
    v,
    tbs,
    land,
    water_arr,
    tb_mask: npt.NDArray[np.bool_],
):

    # main calc_bt_ice() block
    vh37chk = v['vh37'][0] - v['adoff'] + v['vh37'][1] * tbs['v37']

    # Compute radchk1
    is_check1 = tbs['h37'] > vh37chk
    is_h37_lt_rc1 = tbs['h37'] < (v['radslp1'] * tbs['v37'] + v['radoff1'])

    iclen1 = np.sqrt(
        np.square(tbs['v37'] - v['wtp'][0]) + np.square(tbs['h37'] - v['wtp'][1])
    )
    is_iclen1_gt_radlen1 = iclen1 > v['radlen1']
    icpix1 = ret_ic_32(
        tbs['v37'],
        tbs['h37'],
        v['wtp'][0],
        v['wtp'][1],
        v['vh37'][0],
        v['vh37'][1],
        p['missval'],
        p['maxic'],
    )
    icpix1[is_h37_lt_rc1 & is_iclen1_gt_radlen1] = 1.0
    # icpix1[is_h37_lt_rc1 & (iclen1 <= v['radlen1'])]
    is_condition1 = is_h37_lt_rc1 & ~(iclen1 > v['radlen1'])
    icpix1[is_condition1] = iclen1[is_condition1] / v['radlen1']

    # Compute radchk2
    is_v19_lt_rc2 = tbs['v19'] < (v['radslp2'] * tbs['v37'] + v['radoff2'])

    iclen2 = np.sqrt(
        np.square(tbs['v37'] - v['wtp2'][0]) + np.square(tbs['v19'] - v['wtp2'][1])
    )
    is_iclen2_gt_radlen2 = iclen2 > v['radlen2']
    icpix2 = ret_ic_32(
        tbs['v37'],
        tbs['v19'],
        v['wtp2'][0],
        v['wtp2'][1],
        v['v1937'][0],
        v['v1937'][1],
        p['missval'],
        p['maxic'],
    )
    icpix2[is_v19_lt_rc2 & is_iclen2_gt_radlen2] = 1.0
    is_condition2 = is_v19_lt_rc2 & ~is_iclen2_gt_radlen2
    icpix2[is_condition2] = iclen2[is_condition2] / v['radlen2']

    ic = icpix1
    ic[~is_check1] = icpix2[~is_check1]

    is_ic_is_missval = ic == p['missval']
    ic[is_ic_is_missval] = p['missval']
    ic[~is_ic_is_missval] = ic[~is_ic_is_missval] * 100.0

    ic[water_arr == 1] = 0.0
    ic[tb_mask] = p['missval']
    ic[land != 0] = p['landval']

    return ic


if __name__ == '__main__':
    # Set a flag for getting *exactly* the same output file
    # as the Goddard fortran code produces
    do_exact = True
    # do_exact = False

    orig_params: Params = import_cfg_file(THIS_DIR / 'ret_ic_params.json')
    params: Params = import_cfg_file(THIS_DIR / 'ret_ic_params.json')

    orig_vars: Variables = import_cfg_file(THIS_DIR / 'ret_ic_variables.json')
    variables: Variables = import_cfg_file(THIS_DIR / 'ret_ic_variables.json')

    # Convert params to variables
    otbs: dict[str, npt.NDArray[np.float32]] = {}

    for tb in ('v19', 'h37', 'v37', 'v22'):
        otbs[tb] = read_tb_field(
            (
                THIS_DIR / params['raw_fns'][tb]  # type: ignore [literal-required]
            ).resolve()
        )

    land_arr = np.fromfile(
        (THIS_DIR / params['raw_fns']['land']).resolve(), dtype=np.int16
    ).reshape(448, 304)

    tb_mask = tb_data_mask(
        tbs=(
            otbs['v37'],
            otbs['h37'],
            otbs['v19'],
            otbs['v22'],
        ),
        min_tb=params['mintb'],
        max_tb=params['maxtb'],
    )

    # *** compute tbs ***
    # Note: even though the xfer doesn not result in identical fields,
    #       the sample output is still identical (!)
    tbs = xfer_tbs_nrt(
        otbs['v37'], otbs['h37'], otbs['v19'], otbs['v22'], params['sat']
    )

    # *** CALL ret_para_nsb2 for vh37 ***
    para_vals = ret_para_nsb2('vh37', params['sat'], params['seas'])
    params['wintrc'] = para_vals['wintrc']
    params['wslope'] = para_vals['wslope']
    params['wxlimt'] = para_vals['wxlimt']
    params['ln1'] = para_vals['lnline']
    params['lnchk'] = para_vals['lnchk']
    variables['wtp'] = para_vals['wtp']
    variables['itp'] = para_vals['itp']

    # *** CALL ret_water_ssmi() ***
    new_water_arr = ret_water_ssmi(
        tbs['v37'],
        tbs['h37'],
        tbs['v22'],
        tbs['v19'],
        land_arr,
        tb_mask,
        params['wslope'],
        params['wintrc'],
        params['wxlimt'],
        params['ln1'],
    )
    water_arr = new_water_arr

    # Set wtp, which is tp37v and tp37h
    variables['wtp37v'] = ret_wtp_32(water_arr, tbs['v37'])
    variables['wtp37h'] = ret_wtp_32(water_arr, tbs['h37'])

    # assert these keys are not None so the typechecker does not complain.
    assert variables['wtp37v'] is not None
    assert variables['wtp37h'] is not None

    if (variables['wtp'][0] - 10) < variables['wtp37v'] < (variables['wtp'][0] + 10):
        variables['wtp'][0] = variables['wtp37v']
    if (variables['wtp'][1] - 10) < variables['wtp37h'] < (variables['wtp'][1] + 10):
        variables['wtp'][1] = variables['wtp37h']

    # ## CALL ret_linifit2() for vh37 ###
    # Note: I cannot get this to give exact answers because of roundoff
    # The differences are roughly +/- 0.0002 in float32 conc values
    calc_vh37 = ret_linfit_32(
        land_arr,
        tb_mask,
        tbs['v37'],
        tbs['h37'],
        params['ln1'],
        params['lnchk'],
        params['add1'],
        water_arr,
    )
    if do_exact:
        print('\nNot using calculated vh37:')
        print(f'  calculated: {calc_vh37}')
        print(f"        used: {variables['vh37']}")
    else:
        variables['vh37'] = calc_vh37

    # *** CALL ret_adj_adoff ***
    variables['adoff'] = ret_adj_adoff(variables['wtp'], variables['vh37'])

    # *** CALL ret_para_nsb2 for v1937 ***
    para_vals = ret_para_nsb2('v1937', params['sat'], params['seas'])
    params['ln2'] = para_vals['lnline']
    variables['wtp2'] = para_vals['wtp']
    variables['itp2'] = para_vals['itp']

    # variables['wtp2'] = para_vals['wtp']
    variables['itp2'] = para_vals['itp']
    variables['v1937'] = para_vals['iceline']

    # variables['wtp2'] = para_vals['wtp']

    # *** CALL ret_wtp() for wtp19v ***
    variables['wtp19v'] = ret_wtp_32(water_arr, tbs['v19'])

    assert variables['wtp19v'] is not None

    if (variables['wtp2'][0] - 10) < variables['wtp37v'] < (variables['wtp2'][0] + 10):
        variables['wtp2'][0] = variables['wtp37v']
    if (variables['wtp2'][1] - 10) < variables['wtp19v'] < (variables['wtp2'][1] + 10):
        variables['wtp2'][1] = variables['wtp19v']

    # ## CALL ret_linifit2() for v1937 ###
    # Note: I cannot get this to give exact answers because of roundoff
    # The differences are roughly +/- 0.0002 in float32 conc values

    # Try the ret_para... values for v1937
    calc_v1937 = ret_linfit_32(
        land_arr,
        tb_mask,
        tbs['v37'],
        tbs['v19'],
        params['ln2'],
        params['lnchk'],
        params['add2'],
        water_arr,
        tba=tbs['h37'],
        iceline=variables['vh37'],
        adoff=variables['adoff'],
    )
    if do_exact:
        variables['v1937'] = orig_vars['v1937']
        print('\nNot using calculated v1937 values:')
        print(f'   calc v1937: {calc_v1937}')
        print(f"   used v1937: {variables['v1937']}")
    else:
        variables['v1937'] = calc_v1937

    # ## LINES calculating radslp1 ... to radlen2 ###
    variables = calc_rad_coeffs_32(params, variables)

    # ## LINES with loop calling (in part) ret_ic() ###
    iceout = calc_bt_ice(params, variables, tbs, land_arr, water_arr, tb_mask)

    # *** Do sst cleaning ***
    iceout_sst = sst_clean_sb2(
        iceout, params['missval'], params['landval'], params['month'], params['pole']
    )

    # *** Do spatial interp ***
    iceout_spi = spatial_interp(
        iceout_sst,
        params['missval'],
        params['landval'],
        (THIS_DIR / params['raw_fns']['nphole']).resolve(),
    )

    # *** Do spatial interp ***
    iceout_fix = coastal_fix(
        iceout_sst, params['missval'], params['landval'], params['minic']
    )
    iceout_fix[iceout_fix < params['minic']] = 0

    # *** Do fix_output ***
    fixout = fix_output_gdprod(
        iceout_fix,
        params['minval'],
        params['maxval'],
        params['landval'],
        params['missval'],
    )

    # *** Write the output to a similar file name as fortran code ***
    # Derive year from a tb filename
    sat, year, month, day = get_satymd_from_tbfn(params['raw_fns']['v37'])
    ofn = f'NH_{year}{month}{day}_py_NRT_f{sat}.ic'
    # TODO: consider writing this file out to an explicit output dir. Where?
    fixout.tofile(THIS_DIR / ofn)
    print(f'Wrote output file: {ofn}')

    print('Finished compute_bt_ic.py')
    print(' ')  # To add a blank line after run
