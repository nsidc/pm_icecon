"""Compute the Bootstrap ice concentration.

Takes values from part of boot_ice_sb2_ssmi_np_nrt.f
and computes:
    iceout
"""

import calendar
import copy
import datetime as dt
from functools import reduce
from typing import Literal, Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr
from loguru import logger

from pm_icecon.bt._types import Line, Tiepoint
from pm_icecon.config.models.bt import (
    BootstrapParams,
    WeatherFilterParams,
    WeatherFilterParamsForSeason,
)
from pm_icecon.constants import DEFAULT_FLAG_VALUES
from pm_icecon.errors import BootstrapAlgError, UnexpectedSatelliteError


def f(num):
    # return float32 of num
    return np.float32(num)


# TODO: this function very similar to `get_invalid_tbs_mask` in `compute_nt_ic`.
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


def xfer_class_tbs(
    *,
    tb_v37: npt.NDArray,
    tb_h37: npt.NDArray,
    tb_v19: npt.NDArray,
    tb_v22: npt.NDArray,
    sat: Literal['f17', 'f18'],
) -> dict[str, npt.NDArray[np.float32]]:
    """Transform selected CLASS (NRT) TBs for consistentcy with timeseries.

    Some CLASS data should be transformed via linear regression for consistenicy
    with other data sources (TODO: which ones exactly?)

    TODO: make sure this description is...descriptive enough.
    """
    # NRT regressions
    if sat == 'f17':
        tb_v37 = fadd(fmul(1.0170066, tb_v37), -4.9383355)
        tb_h37 = fadd(fmul(1.0009720, tb_h37), -1.3709822)
        tb_v19 = fadd(fmul(1.0140723, tb_v19), -3.4705583)
        tb_v22 = fadd(fmul(0.99652931, tb_v22), -0.82305684)
    elif sat == 'f18':
        tb_v37 = fadd(fmul(1.0104497, tb_v37), -3.3174017)
        tb_h37 = fadd(fmul(0.98914390, tb_h37), 1.2031835)
        tb_v19 = fadd(fmul(1.0057373, tb_v19), -0.92638520)
        tb_v22 = fadd(fmul(0.98793409, tb_v22), 1.2108198)
    else:
        raise UnexpectedSatelliteError(f'No such tb xform: {sat}')

    return {
        'tb_v37': tb_v37,
        'tb_h37': tb_h37,
        'tb_v19': tb_v19,
        'tb_v22': tb_v22,
    }


# TODO: is this function really specific to 37v37h or should it be more generic?
# If specific, also rename `wtp` and rename func to make this clear. If not,
# rename `vh37` kwarg to e.g., 'line'?
def ret_adj_adoff(*, wtp: Tiepoint, line_37v37h: Line, perc=0.92) -> float:
    # replaces ret_adj_adoff()
    # wtp is one water tie point
    wtp_x, wtp_y = f(wtp[0]), f(wtp[1])
    off = line_37v37h['offset']
    slp = line_37v37h['slope']

    x = ((wtp_x / slp) + wtp_y - off) / (slp + 1.0 / slp)
    y = slp * x + off

    dx = wtp_x - x
    dx2 = perc * dx
    x2 = wtp_x - dx2

    dy = y - wtp_y
    dy2 = perc * dy
    y2 = wtp_y + dy2

    new_off = y2 - slp * x2

    adoff = off - new_off

    return adoff


# TODO: rename. This doesn't actually return a wtp, it retuns one of it's terms
# (x or y)
def _ret_wtp_32(
    weather_mask: npt.NDArray[np.bool_],
    tb: npt.NDArray[np.float32],
) -> float:
    """Return either the x or y of the wtp."""
    # Attempt to reproduce Goddard methodology for computing water tie point

    # Note: this *really* should be done with np.percentile()

    pct = 0.02
    n_bins = 1200

    # Compute quarter-Kelvin histograms
    histo, _ = np.histogram(
        tb[weather_mask],
        bins=n_bins,
        range=(0, 300),
    )
    nvals = histo.sum()

    # Remove low-count bins (but don't adjust total ??!!)
    histo[histo <= 10] = 0

    ival = 0
    subtotal = 0
    thresh = f(nvals) * pct
    while (ival < n_bins) and (subtotal < thresh):
        subtotal += histo[ival]
        ival += 1
    ival -= 1  # undo last increment

    # TODO: this expression returns `np.float64`, NOT `np.float32` like `f`
    # returns...
    wtp = f(ival) * 0.25

    return wtp


def get_water_tiepoint(
    *,
    wtp_default: Tiepoint,
    weather_mask: npt.NDArray[np.bool_],
    tbx,
    tby,
) -> Tiepoint:
    """Return the deafult or calculate new wtp.

    If the calculated wtpx and wtpy values are within +/- 10 of the
    `wtp_default`, use the newly calculated values.
    """
    wtpx = _ret_wtp_32(weather_mask, tbx)
    wtpy = _ret_wtp_32(weather_mask, tby)

    new_wtp = list(copy.copy(wtp_default))

    # If the calculated wtps are within the bounds of the default (+/- 10), use
    # the calculated value.
    def _within_plusminus_10(target_value, value) -> bool:
        return (target_value - 10) < value < (target_value + 10)

    if _within_plusminus_10(wtp_default[0], wtpx):
        new_wtp[0] = wtpx
    if _within_plusminus_10(wtp_default[1], wtpy):
        new_wtp[1] = wtpy

    wtp_tuple = (new_wtp[0], new_wtp[1])

    return wtp_tuple


def linfit(xvals, yvals):
    # Implement original Bootstrap linear-fit routine
    # TODO: `np.polyfit(xvals, yvals, 1)` gives the same result to a few
    # decimals. Regression tests pass using this builtin of numpy. Is it worth
    # having this function for complete consistency with the Goddard code?
    nvals = f(xvals.shape[0])
    sumx = np.sum(xvals, dtype=np.float64)
    sumy = np.sum(yvals, dtype=np.float64)
    sumx2 = np.sum(fsqr(xvals), dtype=np.float64)
    sumxy = np.sum(fmul(xvals, yvals), dtype=np.float64)

    delta = (nvals * sumx2) - sumx * sumx
    offset = ((sumx2 * sumy) - (sumx * sumxy)) / delta
    slope = ((sumxy * nvals) - (sumx * sumy)) / delta

    return offset, slope


def ret_linfit_32(
    *,
    land_mask: npt.NDArray[np.bool_],
    tb_mask: npt.NDArray[np.bool_],
    tbx,
    tby,
    lnline: Line,
    add,
    lnchk=1.5,
    weather_mask,
    # If any one of the rest of these arguments is given, the rest must also be
    # non-None. Currently only used for getting the v1937 line, in determining
    # if pixels are valid for use.
    tba=None,
    iceline: Line | None = None,
    adoff=None,
) -> Line:
    # Reproduces both ret_linfit1() and ret_linfit2()
    not_land_or_masked = ~land_mask & ~tb_mask
    if tba is not None and iceline is not None and adoff is not None:
        is_tba_le_modad = tba <= fadd(
            fmul(tbx, iceline['slope']), fsub(iceline['offset'], adoff)
        )
    else:
        is_tba_le_modad = np.full_like(not_land_or_masked, fill_value=True)

    is_tby_gt_lnline = tby > fadd(fmul(tbx, lnline['slope']), lnline['offset'])

    is_valid = not_land_or_masked & is_tba_le_modad & is_tby_gt_lnline & ~weather_mask

    icnt = np.sum(np.where(is_valid, 1, 0))
    if icnt <= 125:
        raise BootstrapAlgError(f'Insufficient valid linfit points: {icnt}')

    xvals = tbx[is_valid].astype(np.float32).flatten().astype(np.float64)
    yvals = tby[is_valid].astype(np.float32).flatten().astype(np.float64)

    intrca, slopeb = linfit(xvals, yvals)

    if slopeb > lnchk:
        raise BootstrapAlgError(
            f'lnchk failed. {slopeb=} > {lnchk=}. '
            'This may need some additional investigation! The code from Goddard would'
            ' fall back on defaults defined by the `iceline` parameter if this'
            ' condition was met. However, it is probably better to investigate'
            ' this situation and determine what to do on a case-by-case basis'
            ' rather than "silently" fall back on some default values. We are not'
            ' sure how the default values of (`iceline`) were originally chosen.'
        )

    fit_off = fadd(intrca, add)
    fit_slp = f(slopeb)
    line = Line(offset=fit_off, slope=fit_slp)

    return line


def ret_ic_32(*, tbx, tby, wtp: Tiepoint, iline: Line, baddata, maxic):
    wtpx = wtp[0]
    wtpy = wtp[1]
    iline_off = iline['offset']
    iline_slp = iline['slope']

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


def rad_adjust_ic(*, ic, tbx, tby, itp: Tiepoint, wtp: Tiepoint, line: Line):
    adjusted_ic = ic.copy()

    radslp2, radoff2, radlen = calc_rad_coeffs(
        itp=itp,
        wtp=wtp,
        line=line,
    )

    is_v19_lt_rc2 = tby < (radslp2 * tbx + radoff2)

    iclen = np.sqrt(np.square(tbx - wtp[0]) + np.square(tby - wtp[1]))
    is_iclen_gt_radlen = iclen > radlen
    adjusted_ic[is_v19_lt_rc2 & is_iclen_gt_radlen] = 1.0
    is_condition = is_v19_lt_rc2 & ~is_iclen_gt_radlen
    adjusted_ic[is_condition] = iclen[is_condition] / radlen

    return adjusted_ic


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


def _get_wx_params(
    *,
    date: dt.date,
    weather_filter_seasons: list[WeatherFilterParamsForSeason],
) -> WeatherFilterParams:
    """Return weather filter params for a given date.

    Given a list of `WeatherFilterParamsForSeason` and a date, return the
    correct weather filter params.

    If a date occurs between seasons, use linear interpolation to determine
    weather filter params from the adjacent seasons.

    TODO: simplify this code! Originally, I thought it would be straightforward
    to simply create a period_range from a season start day/month and season
    end day/month. However, seasons can span the end of the year (e.g., November
    through April).

    This code uses pandas dataframes to build up a list of dates with given
    parameters for each season. Each season has its parameters duplicated for
    all days in the season for the year given by `date` and year + 1. This
    allows pandas to do linear interpolation that occurs 'across' a year.
    """
    monthly_dfs = []
    for season in weather_filter_seasons:

        if season.start_month > season.end_month:
            # E.g., start_month=11 and end_month=4:
            # season_months=[11, 12, 1, 2, 3, 4].
            season_months = list(range(season.start_month, 12 + 1)) + list(
                range(1, season.end_month + 1)
            )

        else:
            season_months = list(range(season.start_month, season.end_month + 1))

        for month in season_months:
            # Default to the start of the month. If we're at the beginning of
            # the season, then optionally use `season.start_day`.
            start_day = 1
            if month == season.start_month:
                start_day = season.start_day if season.start_day else start_day

            # Default to the end of the month. If we're looking at the end of
            # the season, then optionally use `season.end_day`.
            end_day = calendar.monthrange(date.year, month)[1]
            if month == season.end_month:
                end_day = season.end_day if season.end_day else end_day

            periods_this_year = pd.period_range(
                start=pd.Period(year=date.year, month=month, day=start_day, freq='D'),
                end=pd.Period(year=date.year, month=month, day=end_day, freq='D'),
            )

            # if the date we are interested in is in this month of the season,
            # return the weather filter params.
            if pd.Period(date, freq='D') in periods_this_year:
                return season.weather_filter_params

            # Get the same periods for the following year. and include those in
            # the dataframe we are building. This ensures that a date that
            # occurs between seasons that span a year gets correctly
            # interpolated.
            periods_next_year = pd.period_range(
                start=pd.Period(
                    year=date.year + 1, month=month, day=start_day, freq='D'
                ),
                end=pd.Period(year=date.year + 1, month=month, day=end_day, freq='D'),
            )
            all_periods = list(periods_this_year) + list(periods_next_year)

            monthly_dfs.append(
                pd.DataFrame(
                    data={
                        key: [getattr(season.weather_filter_params, key)]
                        * len(all_periods)
                        for key in ('wintrc', 'wslope', 'wxlimt')
                    },
                    index=all_periods,
                )
            )

    # Create a df with a period index that includes an entry for every day so
    # that we can `loc` the date we are interested in.
    df_with_daily_index = pd.DataFrame(
        index=pd.period_range(
            start=pd.Period(year=date.year, month=1, day=1, freq='D'),
            end=pd.Period(year=date.year + 1, month=12, day=31, freq='D'),
        )
    )
    joined = df_with_daily_index.join(pd.concat(monthly_dfs))
    interpolated = joined.interpolate()

    return WeatherFilterParams(
        **{
            key: interpolated.loc[pd.Period(date, freq='D')][key]
            for key in ('wintrc', 'wslope', 'wxlimt')
        }
    )


def get_weather_mask(
    *,
    v37,
    h37,
    v22,
    v19,
    land_mask: npt.NDArray[np.bool_],
    tb_mask: npt.NDArray[np.bool_],
    ln1: Line,
    date: dt.date,
    weather_filter_seasons: list[WeatherFilterParamsForSeason],
) -> npt.NDArray[np.bool_]:
    """Return a water mask that has been weather filtered.

    `True` indicates areas that are water and are weather masked. I.e., `True`
    values should be treated as open ocean.
    """
    season_params = _get_wx_params(
        date=date,
        weather_filter_seasons=weather_filter_seasons,
    )
    wintrc = season_params.wintrc
    wslope = season_params.wslope
    wxlimt = season_params.wxlimt

    # Determine where there is definitely water
    not_land_or_masked = ~land_mask & ~tb_mask
    watchk1 = fadd(fmul(f(wslope), v22), f(wintrc))
    watchk2 = fsub(v22, v19)
    watchk4 = fadd(fmul(ln1['slope'], v37), ln1['offset'])

    is_cond1 = (watchk1 > v19) | (watchk2 > wxlimt)
    # TODO: where does this 230.0 value come from? Should it be configuratble?
    is_cond2 = (watchk4 > h37) | (v37 >= 230.0)

    is_water = not_land_or_masked & is_cond1 & is_cond2

    return is_water


def calc_rad_coeffs(
    *,
    itp: Tiepoint,
    wtp: Tiepoint,
    line: Line,
):
    rad_slope = fdiv(
        fsub(f(itp[1]), f(wtp[1])),
        fsub(f(itp[0]), f(wtp[0])),
    )
    rad_offset = fsub(f(wtp[1]), fmul(f(wtp[0]), f(rad_slope)))
    xint = fdiv(
        fsub(f(rad_offset), f(line['offset'])),
        fsub(f(line['slope']), f(rad_slope)),
    )
    yint = fadd(
        fmul(
            line['slope'],
            f(xint),
        ),
        f(line['offset']),
    )
    rad_len = fsqt(
        fadd(
            fsqr(fsub(f(xint), f(wtp[0]))),
            fsqr(fsub(f(yint), f(wtp[1]))),
        )
    )

    return (rad_slope, rad_offset, rad_len)


def apply_invalid_ice_mask(
    *,
    conc,
    missing_flag_value,
    land_flag_value,
    invalid_ice_mask: npt.NDArray[np.bool_],
):
    """Set all `invalid_ice_mask`ed areas that are not missing or land to 0.

    Implementation of GSFC fortran `sst_clean_sb2()` routine.
    """
    is_not_land = conc != land_flag_value
    is_not_miss = conc != missing_flag_value
    is_not_land_miss_sst = is_not_land & is_not_miss & invalid_ice_mask

    ice_sst = conc.copy()
    ice_sst[is_not_land_miss_sst] = 0.0

    return ice_sst


def coastal_fix(
    *,
    conc: npt.NDArray,
    missing_flag_value,
    land_flag_value,
    minic,
):
    # Apply coastal_fix() routine per Bootstrap

    # Calculate 'temp' array
    #   -1 is no ice
    #    1 is safe from removal
    #    0 is might-be-removed
    temp = np.ones_like(conc, dtype=np.int16)
    is_land_or_lowice = (conc == land_flag_value) | ((conc >= 0) & (conc < minic))
    temp[is_land_or_lowice] = -1

    is_seaice = (conc > 0) & (conc <= 100.0)

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
        rolled_offn1 = np.roll(conc, offp1, axis=(1, 0))  # land
        rolled_off00 = conc.copy()  # .  k1p0 k2p0
        rolled_offp1 = np.roll(conc, offn1, axis=(1, 0))  # k1 k2p1
        rolled_offp2 = np.roll(conc, offn2, axis=(1, 0))  # k2

        # is_rolled_land = rolled_offn1 == land_flag_value
        is_rolled_land = rolled_offn1 == land_flag_value

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
            & (rolled_off00 != missing_flag_value)
            & (rolled_off00 != land_flag_value)
        )
        is_k2p0 = (
            (is_k2)
            & (rolled_off00 > 0)
            & (rolled_off00 != missing_flag_value)
            & (rolled_off00 != land_flag_value)
        )
        is_k2p1 = (
            (is_k2)
            & (rolled_offp1 > 0)
            & (rolled_offp1 != missing_flag_value)
            & (rolled_offp1 != land_flag_value)
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
            logger.debug('Fixing out of bounds error in `coastal_fix`')
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

    # Calculate 'conc2' array
    # This is initially a copy of the conc array, but then has values
    #   set to zero where 'appropriate' based on the temp array
    conc2 = conc.copy()

    # This is very complicated to figure out as modification
    # of the series of off_sets.  Simply coding each of the
    # four change sections manually

    # Note: some of these conditional arrays might be set more than 1x

    # Compute shifted conc grid, for land check
    land_check = np.roll(conc, (0, 1), axis=(1, 0))  # land check
    is_rolled_land = land_check == land_flag_value

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

    # Changing conc2(i,j+1) to 0
    locs_ip0jp1 = np.where(
        is_considered & is_tip1jp1_lt0 & is_tim1jp1_lt0 & is_tip0jp1_eq0
    )
    change_locs_conc2_ip0jp1 = tuple([locs_ip0jp1[0] + 1, locs_ip0jp1[1] + 0])
    conc2[change_locs_conc2_ip0jp1] = 0

    # Changing conc2(i,j) to 0
    locs_ip0jp0 = np.where(
        is_considered
        & is_tip1jp1_lt0
        & is_tim1jp1_lt0
        & is_tip1jp0_lt0
        & is_tim1jp0_lt0
    )
    change_locs_conc2_ip0jp0 = tuple([locs_ip0jp0[0], locs_ip0jp0[1]])
    conc2[change_locs_conc2_ip0jp0] = 0

    # Second conc2 change section

    # Compute shifted conc grid, for land check
    land_check = np.roll(conc, (0, -1), axis=(1, 0))  # land check
    is_rolled_land = land_check == land_flag_value

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

    # Changing conc2(i,j-1) to 0
    locs_ip0jm1 = np.where(
        is_considered & is_tip1jm1_le0 & is_tim1jm1_le0 & is_tip0jm1_eq0
    )
    change_locs_conc2_ip0jm1 = tuple([locs_ip0jm1[0] - 1, locs_ip0jm1[1] + 0])
    conc2[change_locs_conc2_ip0jm1] = 0

    # Changing conc2(i,j) to 0
    locs_ip0jp0 = np.where(
        is_considered
        & is_tip1jm1_le0
        & is_tim1jm1_le0
        & is_tip1jp0_le0
        & is_tim1jp0_le0
    )
    change_locs_conc2_ip0jp0 = tuple([locs_ip0jp0[0], locs_ip0jp0[1]])
    conc2[change_locs_conc2_ip0jp0] = 0

    # Third conc2 change section

    # Compute shifted conc grid, for land check
    land_check = np.roll(conc, (1, 0), axis=(1, 0))
    is_rolled_land = land_check == land_flag_value

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

    # Changing conc2(i+1,j) to 0
    locs_ip1jp0 = np.where(
        is_considered & is_tip1jp1_le0 & is_tip1jp1_le0 & is_tip1jp0_eq0
    )
    change_locs_conc2_ip1jp0 = tuple([locs_ip1jp0[0] + 0, locs_ip1jp0[1] + 1])
    conc2[change_locs_conc2_ip1jp0] = 0

    # Changing conc2(i,j) to 0
    locs_ip0jp0 = np.where(
        is_considered
        & is_tip1jp1_le0
        & is_tip1jp1_le0
        & is_tip0jm1_le0
        & is_tip0jp1_le0
    )
    change_locs_conc2_ip0jp0 = tuple([locs_ip0jp0[0], locs_ip0jp0[1]])
    conc2[change_locs_conc2_ip0jp0] = 0

    # Fourth section

    # Compute shifted conc grid, for land check
    land_check = np.roll(conc, (-1, 0), axis=(1, 0))
    is_rolled_land = land_check == land_flag_value

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

    # Changing conc2(i-1,j) to 0
    locs_im1jp0 = np.where(
        is_considered & is_tim1jm1_le0 & is_tim1jp1_le0 & is_tim1jp0_eq0
    )
    change_locs_conc2_im1jp0 = tuple([locs_im1jp0[0] + 0, locs_im1jp0[1] - 1])
    conc2[change_locs_conc2_im1jp0] = 0

    # Changing conc2(i,j) to 0
    locs_ip0jp0 = np.where(
        is_considered
        & is_tim1jm1_le0
        & is_tim1jp1_le0
        & is_tip0jm1_le0
        & is_tip0jp1_le0
    )
    change_locs_conc2_ip0jp0 = tuple([locs_ip0jp0[0], locs_ip0jp0[1]])
    conc2[change_locs_conc2_ip0jp0] = 0

    return conc2


def _calc_frac_conc_for_tbset(
    *,
    tbx,
    tby,
    wtp: Tiepoint,
    itp: Tiepoint,
    line: Line,
    missing_data_value: float | int,
    maxic,
):
    """Return fractional sea ice concentration for the given parameters."""
    icpix1 = ret_ic_32(
        tbx=tbx,
        tby=tby,
        wtp=wtp,
        iline=line,
        baddata=missing_data_value,
        maxic=maxic,
    )

    icpix1 = rad_adjust_ic(
        ic=icpix1,
        tbx=tbx,
        tby=tby,
        itp=itp,
        wtp=wtp,
        line=line,
    )

    return icpix1


def calc_bootstrap_conc(
    *,
    maxic_frac,
    line_37v37h: Line,
    adoff,
    line_37v19v: Line,
    wtp_37v37h: Tiepoint,
    wtp_37v19v: Tiepoint,
    itp_37v37h: Tiepoint,
    itp_37v19v: Tiepoint,
    tb_v37: npt.NDArray,
    tb_h37: npt.NDArray,
    tb_v19: npt.NDArray,
    # TODO: can/should we just use `nan`?
    missval: float | int,
):
    """Return a sea ice concentration estimate at every grid cell.

    Concentrations are given as percentage (0-100%).
    """
    # ic_frac == 'ice concentration fraction'
    ic_frac_37v37h = _calc_frac_conc_for_tbset(
        tbx=tb_v37,
        tby=tb_h37,
        wtp=wtp_37v37h,
        itp=itp_37v37h,
        line=line_37v37h,
        missing_data_value=missval,
        maxic=maxic_frac,
    )

    ic_frac_37v19v = _calc_frac_conc_for_tbset(
        tbx=tb_v37,
        tby=tb_v19,
        wtp=wtp_37v19v,
        itp=itp_37v19v,
        line=line_37v19v,
        missing_data_value=missval,
        maxic=maxic_frac,
    )

    ic_frac = ic_frac_37v37h.copy()

    vh37chk = line_37v37h['offset'] - adoff + line_37v37h['slope'] * tb_v37
    is_check1 = tb_h37 > vh37chk
    ic_frac[~is_check1] = ic_frac_37v19v[~is_check1]

    # convert fractional sea ice concentrations to percentages
    ic_perc = ic_frac.copy()
    is_ic_frac_is_missval = ic_frac == missval
    ic_perc[~is_ic_frac_is_missval] = ic_frac[~is_ic_frac_is_missval] * 100.0

    return ic_perc


def goddard_bootstrap(
    *,
    tb_v37: npt.NDArray,
    tb_h37: npt.NDArray,
    tb_v19: npt.NDArray,
    tb_v22: npt.NDArray,
    params: BootstrapParams,
    date: dt.date,
    missing_flag_value: float | int = DEFAULT_FLAG_VALUES.missing,
) -> xr.Dataset:
    """Bootstrap algorithm as organized by the orignal code from GSFC."""
    tb_mask = tb_data_mask(
        tbs=(
            tb_v37,
            tb_h37,
            tb_v19,
            tb_v22,
        ),
        min_tb=params.mintb,
        max_tb=params.maxtb,
    )

    weather_mask = get_weather_mask(
        v37=tb_v37,
        h37=tb_h37,
        v22=tb_v22,
        v19=tb_v19,
        land_mask=params.land_mask,
        tb_mask=tb_mask,
        ln1=params.vh37_params.lnline,
        date=date,
        weather_filter_seasons=params.weather_filter_seasons,
    )

    line_37v37h = ret_linfit_32(
        land_mask=params.land_mask,
        tb_mask=tb_mask,
        tbx=tb_v37,
        tby=tb_h37,
        lnline=params.vh37_params.lnline,
        add=params.add1,
        weather_mask=weather_mask,
    )

    wtp_37v37h = get_water_tiepoint(
        wtp_default=params.vh37_params.water_tie_point,
        weather_mask=weather_mask,
        tbx=tb_v37,
        tby=tb_h37,
    )

    wtp_37v19v = get_water_tiepoint(
        wtp_default=params.v1937_params.water_tie_point,
        weather_mask=weather_mask,
        tbx=tb_v37,
        tby=tb_v19,
    )

    adoff = ret_adj_adoff(wtp=wtp_37v37h, line_37v37h=line_37v37h)

    line_37v19v = ret_linfit_32(
        land_mask=params.land_mask,
        tb_mask=tb_mask,
        tbx=tb_v37,
        tby=tb_v19,
        lnline=params.v1937_params.lnline,
        add=params.add2,
        weather_mask=weather_mask,
        tba=tb_h37,
        iceline=line_37v37h,
        adoff=adoff,
    )

    conc = calc_bootstrap_conc(
        maxic_frac=params.maxic,
        line_37v37h=line_37v37h,
        adoff=adoff,
        line_37v19v=line_37v19v,
        wtp_37v37h=wtp_37v37h,
        wtp_37v19v=wtp_37v19v,
        itp_37v37h=params.vh37_params.ice_tie_point,
        itp_37v19v=params.v1937_params.ice_tie_point,
        tb_v37=tb_v37,
        tb_h37=tb_h37,
        tb_v19=tb_v19,
        missval=missing_flag_value,
    )

    # Apply masks and flag values
    conc[weather_mask] = 0.0
    conc[tb_mask] = 0.0
    conc[params.land_mask] = DEFAULT_FLAG_VALUES.land

    conc = apply_invalid_ice_mask(
        conc=conc,
        missing_flag_value=missing_flag_value,
        land_flag_value=DEFAULT_FLAG_VALUES.land,
        invalid_ice_mask=params.invalid_ice_mask,
    )

    conc = coastal_fix(
        conc=conc,
        missing_flag_value=missing_flag_value,
        land_flag_value=DEFAULT_FLAG_VALUES.land,
        minic=params.minic,
    )
    conc[conc < params.minic] = 0

    ds = xr.Dataset({'conc': (('y', 'x'), conc)})

    return ds
