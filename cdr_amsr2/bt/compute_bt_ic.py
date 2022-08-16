"""Compute the Bootstrap ice concentration.

Takes values from part of boot_ice_sb2_ssmi_np_nrt.f
and computes:
    iceout
"""

import datetime as dt
from functools import reduce
from pathlib import Path
from typing import Literal, Optional, Sequence, get_args

import numpy as np
import numpy.typing as npt
import xarray as xr
from loguru import logger

from cdr_amsr2._types import Hemisphere, ValidSatellites
from cdr_amsr2.bt._types import ParaVals, Variables
from cdr_amsr2.config.models.bt import BootstrapParams
from cdr_amsr2.errors import BootstrapAlgError, UnexpectedSatelliteError
from cdr_amsr2.masks import get_ps_valid_ice_mask

THIS_DIR = Path(__file__).parent


def f(num):
    # return float32 of num
    return np.float32(num)


def tb_data_mask(
    *,
    tbs: Sequence[npt.NDArray[np.float32]],
    min_tb: float,
    max_tb: float,
) -> npt.NDArray[np.bool_]:
    """Return a boolean ndarray inidcating areas of bad data.

    Bad data are locations where any of the given Tbs are outside the range
    defined by (mintb, maxtb).

    NaN values are also considered 'bad' data.

    True values indicate bad data that should be masked. False values indicate
    good data.
    """

    def _is_outofrange_tb(tb, min_tb, max_tb):
        return np.isnan(tb) | (tb < min_tb) | (tb > max_tb)

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
    elif sat == 'u2':
        print(f'No TB modifications for sat: {sat}')
    elif sat == 'a2l1c':
        print(f'No TB modifications for sat: {sat}')
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


def _get_params_for_season(*, sat: str, date: dt.date, hemisphere: Hemisphere):
    is_june_through_oct15 = (date.month >= 6 and date.month <= 9) or (
        date.month == 10 and date.day <= 15
    )

    # Set wintrc, wslope, wxlimt
    if sat == '00':
        if is_june_through_oct15:
            wintrc = 60.1667
            wslope = 0.633333
            wxlimt = 24.00
        else:
            wintrc = 53.4153
            wslope = 0.661017
            wxlimt = 22.00
    elif sat == 'u2':
        # TODO: do we need to implement the seasons 2 and 3 values for AMSR?
        if hemisphere == 'north':
            if is_june_through_oct15:
                # Using the "Season 3" values from ret_parameters_amsru2.f
                wintrc = 82.71
                wslope = 0.5352
                wxlimt = 23.34
            else:
                wintrc = 84.73
                wslope = 0.5352
                wxlimt = 18.39
        else:
            # southern hemisphere has no seasonality
            wintrc = 85.13
            wslope = 0.5379
            wxlimt = 18.596

    elif sat == 'a2l1c':
        if is_june_through_oct15:
            # Using the "Season 3" values from ret_parameters_amsru2.f
            wintrc = 82.71
            wslope = 0.5352
            wxlimt = 23.34
        else:
            wintrc = 84.73
            wslope = 0.5352
            wxlimt = 18.39
    elif sat in get_args(ValidSatellites):
        logger.warning(
            f'Using default seasonal values for {sat}. '
            'You may want to consider defining satellite-specific parameters!'
        )
        if is_june_through_oct15:
            if sat != '17' and sat != '18':
                wintrc = 89.3316
                wslope = 0.501537
            else:
                wintrc = 89.2000
                wslope = 0.503750
            wxlimt = 21.00
        else:
            if sat != '17' and sat != '18':
                wintrc = 90.3355
                wslope = 0.501537
            else:
                wintrc = 87.6467
                wslope = 0.517333
            wxlimt = 14.00
    else:
        raise NotImplementedError(f'No params defined for {sat}')

    return {
        'wintrc': wintrc,
        'wslope': wslope,
        'wxlimt': wxlimt,
    }


def ret_para_nsb2(
    tbset: Literal['vh37', 'v1937'], sat: str, date: dt.date, hemisphere: Hemisphere
) -> ParaVals:
    # TODO: what does this do and why?
    # reproduce effect of ret_para_nsb2()
    # Note: instead of '1' or '2', use description of axes tb1 and tb2
    #       to identify the TB set whose parameters are being set
    #       So, tbset is 'v1937' or 'vh37'
    # Note: 'sat' is a *string*, not an integer

    if hemisphere == 'south' and sat != 'u2':
        raise NotImplementedError('Southern hemisphere is only implemented for AMSR2')

    print(f'in ret_para_nsb2(): sat is {sat}')
    season_params = _get_params_for_season(sat=sat, date=date, hemisphere=hemisphere)

    if sat == 'u2':
        # Values for AMSRU
        print(f'Setting sat values for: {sat}')
        if hemisphere == 'north':
            if tbset == 'vh37':
                wtp = [207.2, 131.9]
                itp = [256.3, 241.2]
                lnline = [-71.99, 1.20]
                iceline = [-30.26, 1.0564]
                lnchk = 1.5
            elif tbset == 'v1937':
                wtp = [207.2, 182.4]
                itp = [256.3, 258.9]
                lnline = [48.26, 0.8048]
                iceline = [110.03, 0.5759]
                lnchk = 1.5
        else:
            if tbset == 'vh37':
                wtp = [207.6, 131.9]
                itp = [259.4, 247.3]
                lnline = [-90.62, 1.2759]
                iceline = [-38.31, 1.0969]
                lnchk = 1.5
            elif tbset == 'v1937':
                wtp = [207.6, 182.7]
                itp = [259.4, 261.6]
                lnline = [62.89, 0.7618]
                iceline = [114.26, 0.5817]
                lnchk = 1.5

    elif sat == 'a2l1c':
        # Values for AMSRU
        print(f'Setting sat values for: {sat}')
        if tbset == 'vh37':
            wtp = [207.2, 131.9]
            itp = [256.3, 241.2]
            lnline = [-71.99, 1.20]
            iceline = [-30.26, 1.0564]
            lnchk = 1.5
        elif tbset == 'v1937':
            wtp = [207.2, 182.4]
            itp = [256.3, 258.9]
            lnline = [48.26, 0.8048]
            iceline = [110.03, 0.5759]
            lnchk = 1.5
    else:
        # Values for DMSP
        print(f'Setting sat values for: {sat}')
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
        **season_params,  # type: ignore
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
    histo, _ = np.histogram(
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
    land_mask: npt.NDArray[np.bool_],
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

    not_land_or_masked = ~land_mask & ~tb_mask
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


def fadd(a: npt.ArrayLike, b: npt.ArrayLike):
    return np.add(a, b, dtype=np.float32)


def fsub(a: npt.ArrayLike, b: npt.ArrayLike):
    return np.subtract(a, b, dtype=np.float32)


def fmul(a: npt.ArrayLike, b: npt.ArrayLike):
    return np.multiply(a, b, dtype=np.float32)


def fdiv(a: npt.ArrayLike, b: npt.ArrayLike):
    return np.divide(a, b, dtype=np.float32)


def fsqr(a: npt.ArrayLike):
    return np.square(a, dtype=np.float32)


def fsqt(a: npt.ArrayLike):
    return np.sqrt(a, dtype=np.float32)


def ret_water_ssmi(
    v37,
    h37,
    v22,
    v19,
    land_mask: npt.NDArray[np.bool_],
    tb_mask: npt.NDArray[np.bool_],
    wslope,
    wintrc,
    wxlimt,
    ln1,
) -> npt.NDArray[np.int16]:
    # Determine where there is definitely water
    not_land_or_masked = ~land_mask & ~tb_mask
    watchk1 = fadd(fmul(f(wslope), v22), f(wintrc))
    watchk2 = fsub(v22, v19)
    watchk4 = fadd(fmul(ln1[1], v37), ln1[0])

    is_cond1 = (watchk1 > v19) | (watchk2 > wxlimt)
    is_cond2 = (watchk4 > h37) | (v37 >= 230.0)

    is_water = not_land_or_masked & is_cond1 & is_cond2

    water = np.zeros_like(land_mask, dtype=np.int16)
    water[is_water] = 1

    return water


def calc_rad_coeffs_32(v: Variables):
    # Compute radlsp, radoff, radlen vars
    v_out = v.copy()

    v_out['radslp1'] = fdiv(
        fsub(f(v_out['itp'][1]), f(v_out['wtp'][1])),
        fsub(f(v_out['itp'][0]), f(v_out['wtp'][0])),
    )
    v_out['radoff1'] = fsub(
        f(v_out['wtp'][1]), fmul(f(v_out['wtp'][0]), f(v_out['radslp1']))
    )
    xint = fdiv(
        fsub(f(v_out['radoff1']), f(v_out['vh37'][0])),
        fsub(f(v_out['vh37'][1]), f(v_out['radslp1'])),
    )
    yint = fadd(fmul(v_out['vh37'][1], f(xint)), f(v_out['vh37'][0]))
    v_out['radlen1'] = fsqt(
        fadd(
            fsqr(fsub(f(xint), f(v_out['wtp'][0]))),
            fsqr(fsub(f(yint), f(v_out['wtp'][1]))),
        )
    )

    v_out['radslp2'] = fdiv(
        fsub(f(v_out['itp2'][1]), f(v_out['wtp2'][1])),
        fsub(f(v_out['itp2'][0]), f(v_out['wtp2'][0])),
    )
    v_out['radoff2'] = fsub(
        f(v_out['wtp2'][1]), fmul(f(v_out['wtp2'][0]), f(v_out['radslp2']))
    )
    xint = fdiv(
        fsub(f(v_out['radoff2']), f(v_out['v1937'][0])),
        fsub(f(v_out['v1937'][1]), f(v_out['radslp2'])),
    )
    yint = fadd(fmul(f(v_out['v1937'][1]), f(xint)), f(v_out['v1937'][0]))
    v_out['radlen2'] = fsqt(
        fadd(
            fsqr(fsub(f(xint), f(v_out['wtp2'][0]))),
            fsqr(fsub(f(yint), f(v_out['wtp2'][1]))),
        )
    )

    return v_out


def sst_clean_sb2(
    *,
    sat: ValidSatellites,
    iceout,
    missval,
    landval,
    date: dt.date,
    hemisphere: Hemisphere,
    resolution: str,
):
    # implement fortran's sst_clean_sb2() routine

    sst_mask: npt.NDArray[np.uint8 | np.int16]

    if sat == 'a2l1c':
        # NOTE: E2N == EASE2 North
        print('Reading valid ice mask for E2N 6.25km grid')
        sst_fn = Path(
            f'/share/apps/amsr2-cdr/bootstrap_masks/valid_seaice_e2n6.25_{date:%m}.dat'
        )
        sst_mask = np.fromfile(sst_fn, dtype=np.uint8).reshape(1680, 1680)
        is_high_sst = sst_mask == 50
    else:
        is_high_sst = get_ps_valid_ice_mask(
            hemisphere=hemisphere,
            date=date,
            resolution=resolution,  # type: ignore[arg-type]
        )

    is_not_land = iceout != landval
    is_not_miss = iceout != missval
    is_not_land_miss_sst = is_not_land & is_not_miss & is_high_sst

    ice_sst = iceout.copy()
    ice_sst[is_not_land_miss_sst] = 0.0

    return ice_sst


def spatial_interp(
    sat,  # TODO: type of 'sat'
    ice: npt.NDArray[np.float32],  # TODO: conc?
    missval: float,
    landval: float,
    pole_mask: Optional[npt.NDArray[np.bool_]],
) -> npt.NDArray[np.float32]:
    iceout = ice.copy()
    # implement fortran's spatial_interp() routine
    # Use -200 as a not-valid ocean sentinel value
    # so that it works with np.roll
    oceanvals = iceout.copy()

    total = np.zeros_like(oceanvals, dtype=np.float32)
    count = np.zeros_like(oceanvals, dtype=np.int32)
    for joff in range(-1, 2):
        for ioff in range(-1, 2):
            # TODO: consider using `scipy.ndimage.shift` instead of `np.roll`
            # here and elsewhere in the code.
            rolled = np.roll(oceanvals, (joff, ioff), axis=(1, 0))
            not_land_nor_miss = (rolled != landval) & (rolled != missval)
            total[not_land_nor_miss] += rolled[not_land_nor_miss]
            count[not_land_nor_miss] += 1

    count[count == 0] = 1
    replace_vals = fdiv(total, count)

    replace_locs = (oceanvals == missval) & (count >= 1)

    if pole_mask is not None:
        replace_locs = replace_locs & ~pole_mask

    iceout[replace_locs] = replace_vals[replace_locs]

    # Now, replace pole if e2n6.25
    if sat == 'a2l1c':
        # TODO: This pole hole function needs some work(!)
        print('Setting pole hole for a2l1c')

        iceout_nearpole = iceout[820:860, 820:860]

        is_pole = iceout_nearpole == 0

        iceout_nearpole[is_pole] = 110

        print(f'Replaced {np.sum(np.where(is_pole, 1, 0))} values at pole')

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

    off_set = (
        np.array((0, 1)),
        np.array((0, -1)),
        np.array((1, 0)),
        np.array((-1, 0)),
    )

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
        try:
            temp[change_locs_k2p1] = 0
        except IndexError:
            print('Fixing out of bounds error')
            locs0 = change_locs_k2p1[0]
            locs1 = change_locs_k2p1[1]

            where_bad_0 = np.where(locs0 == 1680)
            # TODO: should we keep this variable around?
            # where_bad_1 = np.where(locs1 == 1680)

            new_locs0 = np.delete(locs0, where_bad_0)
            new_locs1 = np.delete(locs1, where_bad_0)

            change_locs_k2p1 = tuple([new_locs0, new_locs1])

            try:
                temp[change_locs_k2p1] = 0
            except IndexError:
                raise RuntimeError('Could not fix Index Error')

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


def fix_output_gdprod(conc, minval, maxval, landval, missval) -> npt.NDArray[np.int16]:
    """Scale the given concentration field by 10.

    TODO:
      * Do we want to scale the output by 10? If not, get rid of this? Maybe
        make this optional?
      * Is there ever a case where the valid conc range isn't going to be 0-100?
        If we just need to set neg values to 0 and cap out concs at 100, then we
        can get rid of the 'minval' and 'maxval' parameters. This is the only
        place they're used.
    """
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


def calc_bt_ice(
    p: BootstrapParams,
    v: Variables,
    tbs,
    land_mask: npt.NDArray[np.bool_],
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
        p.missval,
        p.maxic,
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
        p.missval,
        p.maxic,
    )
    icpix2[is_v19_lt_rc2 & is_iclen2_gt_radlen2] = 1.0
    is_condition2 = is_v19_lt_rc2 & ~is_iclen2_gt_radlen2
    icpix2[is_condition2] = iclen2[is_condition2] / v['radlen2']

    ic = icpix1
    ic[~is_check1] = icpix2[~is_check1]

    is_ic_is_missval = ic == p.missval
    ic[is_ic_is_missval] = p.missval
    ic[~is_ic_is_missval] = ic[~is_ic_is_missval] * 100.0

    ic[water_arr == 1] = 0.0
    ic[tb_mask] = p.missval
    ic[land_mask] = p.landval

    return ic


def bootstrap(
    *,
    tbs: dict[str, npt.NDArray],
    params: BootstrapParams,
    variables: Variables,
    date: dt.date,
    hemisphere: Hemisphere,
    # TODO: should be grid-independent. We should probably pass in the valid ice
    # mask like we do for the pole hole and land mask via `params`
    resolution: str,
) -> xr.Dataset:
    """Run the boostrap algorithm."""
    tb_mask = tb_data_mask(
        tbs=(
            tbs['v37'],
            tbs['h37'],
            tbs['v19'],
            tbs['v22'],
        ),
        min_tb=params.mintb,
        max_tb=params.maxtb,
    )

    tbs = xfer_tbs_nrt(tbs['v37'], tbs['h37'], tbs['v19'], tbs['v22'], params.sat)

    para_vals_vh37 = ret_para_nsb2('vh37', params.sat, date, hemisphere)
    wintrc = para_vals_vh37['wintrc']
    wslope = para_vals_vh37['wslope']
    wxlimt = para_vals_vh37['wxlimt']
    ln1 = para_vals_vh37['lnline']
    lnchk = para_vals_vh37['lnchk']
    variables['wtp'] = para_vals_vh37['wtp']
    variables['itp'] = para_vals_vh37['itp']

    water_arr = ret_water_ssmi(
        tbs['v37'],
        tbs['h37'],
        tbs['v22'],
        tbs['v19'],
        params.land_mask,
        tb_mask,
        wslope,
        wintrc,
        wxlimt,
        ln1,
    )

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

    calc_vh37 = ret_linfit_32(
        params.land_mask,
        tb_mask,
        tbs['v37'],
        tbs['h37'],
        ln1,
        lnchk,
        params.add1,
        water_arr,
    )
    variables['vh37'] = calc_vh37

    variables['adoff'] = ret_adj_adoff(variables['wtp'], variables['vh37'])

    para_vals_v1937 = ret_para_nsb2('v1937', params.sat, date, hemisphere)
    ln2 = para_vals_v1937['lnline']
    variables['wtp2'] = para_vals_v1937['wtp']
    variables['itp2'] = para_vals_v1937['itp']
    variables['v1937'] = para_vals_v1937['iceline']

    variables['wtp19v'] = ret_wtp_32(water_arr, tbs['v19'])

    assert variables['wtp19v'] is not None

    if (variables['wtp2'][0] - 10) < variables['wtp37v'] < (variables['wtp2'][0] + 10):
        variables['wtp2'][0] = variables['wtp37v']
    if (variables['wtp2'][1] - 10) < variables['wtp19v'] < (variables['wtp2'][1] + 10):
        variables['wtp2'][1] = variables['wtp19v']

    # Try the ret_para... values for v1937
    calc_v1937 = ret_linfit_32(
        params.land_mask,
        tb_mask,
        tbs['v37'],
        tbs['v19'],
        ln2,
        lnchk,
        params.add2,
        water_arr,
        tba=tbs['h37'],
        iceline=variables['vh37'],
        adoff=variables['adoff'],
    )
    variables['v1937'] = calc_v1937

    # ## LINES calculating radslp1 ... to radlen2 ###
    variables = calc_rad_coeffs_32(variables)

    # ## LINES with loop calling (in part) ret_ic() ###
    iceout = calc_bt_ice(params, variables, tbs, params.land_mask, water_arr, tb_mask)

    # *** Do sst cleaning ***
    print(f'before sst_clean, params:\n{params}')
    iceout_sst = sst_clean_sb2(
        sat=params.sat,
        iceout=iceout,
        missval=params.missval,
        landval=params.landval,
        date=date,
        hemisphere=hemisphere,
        resolution=resolution,
    )

    # *** Do spatial interp ***
    iceout_sst = spatial_interp(
        params.sat,
        iceout_sst,
        params.missval,
        params.landval,
        params.pole_mask,
    )

    # *** Do spatial interp ***
    iceout_fix = coastal_fix(iceout_sst, params.missval, params.landval, params.minic)
    iceout_fix[iceout_fix < params.minic] = 0

    # *** Do fix_output ***
    fixout = fix_output_gdprod(
        iceout_fix,
        params.minval,
        params.maxval,
        params.landval,
        params.missval,
    )

    ds = xr.Dataset({'conc': (('y', 'x'), fixout)})

    return ds
