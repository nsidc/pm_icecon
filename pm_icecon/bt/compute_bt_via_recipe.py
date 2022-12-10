"""Compute the Bootstrap ice concentration using a recipe

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

from pm_icecon._types import Hemisphere
from pm_icecon.bt._types import Tiepoint
from pm_icecon.config.models.bt import (
    BootstrapParams,
    WeatherFilterParams,
    WeatherFilterParamsForSeason,
)
from pm_icecon.constants import DEFAULT_FLAG_VALUES
from pm_icecon.errors import BootstrapAlgError, UnexpectedSatelliteError

from pm_icecon.fetch.au_si import AU_SI_RESOLUTIONS, get_au_si_tbs

from pm_icecon.bt.masks import get_ps_invalid_ice_mask
from pm_icecon.masks import (
    get_e2n625_land_mask,
    get_ps_land_mask,
    get_ps_pole_hole_mask,
)
from pm_icecon.interpolation import spatial_interp_tbs
from pm_icecon.bt.params.amsr2 import AMSR2_NORTH_PARAMS, AMSR2_SOUTH_PARAMS


def get_standard_bootstrap_recipe():
    """Return a dictionary of the standard recipe for AU_SI12 bootstrap"""
    bt_recipe = {}

    bt_recipe['run_parameters'] = {
        'gridid': 'psn12.5',
        'date': dt.date(2020, 1, 1),
    }

    bt_recipe['tb_parameters'] = {
        'tb_source': 'au_si12',
        'mintb': 10.0,
        'maxtb': 320.0,
    }

    bt_recipe['bootstrap_parameters'] = {
        'add1': 0.0,
        'add2': -2.0,
        'minic': 10.0,
        'maxic': 1.0,
        'maxtb': 320.0,
        'vh37_params': {
            'water_tie_point': None,
            'ice_tie_point':None,
            'lnline': (None, None),
        },
        'v1937_params': {
            'water_tie_point': None,
            'ice_tie_point':None,
            'lnline': (None, None),
        },
    }

    bt_recipe['ancillary_sources'] = {
        'surface_mask': 'default',
        'pole_mask': 'default',
        'weather_filters': ('bt_22v',),
        'valid_ice_mask': 'bt_monthly',
        'land_spillover': ('bt', '75km'),
    }

    return bt_recipe

                  
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


def ret_adj_adoff(*, wtp: Tiepoint, vh37: list[float], perc=0.92) -> float:
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


def ret_wtp_32(
    water_mask: npt.NDArray[np.bool_],
    tb: npt.NDArray[np.float32],
) -> float:
    # Attempt to reproduce Goddard methodology for computing water tie point

    # Note: this *really* should be done with np.percentile()

    pct = 0.02

    # Compute quarter-Kelvin histograms
    histo, _ = np.histogram(
        tb[water_mask],
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


def get_water_tiepoints(
    *,
    water_mask,
    tb_v37,
    tb_h37,
    tb_v19,
    wtp1_default: Tiepoint,
    wtp2_default: Tiepoint,
) -> tuple[Tiepoint, Tiepoint]:
    def _within_plusminus_10(target_value, value) -> bool:
        return (target_value - 10) < value < (target_value + 10)

    # Get wtp1
    wtp1 = list(copy.copy(wtp1_default))

    wtp37v = ret_wtp_32(water_mask, tb_v37)
    wtp37h = ret_wtp_32(water_mask, tb_h37)

    # If the calculated wtps are within the bounds of the default (+/- 10), use
    # the calculated value.
    if _within_plusminus_10(wtp1_default[0], wtp37v):
        wtp1[0] = wtp37v
    if _within_plusminus_10(wtp1_default[1], wtp37h):
        wtp1[1] = wtp37h

    # get wtp2
    wtp2 = list(copy.copy(wtp2_default))

    wtp19v = ret_wtp_32(water_mask, tb_v19)

    # If the calculated wtps are within the bounds of the default (+/- 10), use
    # the calculated value.
    if (wtp2_default[0] - 10) < wtp37v < (wtp2_default[0] + 10):
        wtp2[0] = wtp37v
    if (wtp2_default[1] - 10) < wtp19v < (wtp2_default[1] + 10):
        wtp2[1] = wtp19v

    water_tiepoints: tuple[Tiepoint, Tiepoint] = (  # type: ignore[assignment]
        tuple(wtp1),
        tuple(wtp2),
    )

    return water_tiepoints


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
    *,
    land_mask: npt.NDArray[np.bool_],
    tb_mask: npt.NDArray[np.bool_],
    tbx,
    tby,
    lnline,
    add,
    lnchk=1.5,
    water_mask,
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

    is_valid = not_land_or_masked & is_tba_le_modad & is_tby_gt_lnline & ~water_mask

    icnt = np.sum(np.where(is_valid, 1, 0))
    if icnt <= 125:
        raise BootstrapAlgError(f'Insufficient valid linfit points: {icnt}')

    xvals = tbx[is_valid].astype(np.float32).flatten().astype(np.float64)
    yvals = tby[is_valid].astype(np.float32).flatten().astype(np.float64)

    intrca, slopeb = linfit_32(xvals, yvals)

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


# TODO: change the name of this function. Or, do we need different conditions
# for non SSMI data?
def ret_water_ssmi(
    *,
    v37,
    h37,
    v22,
    v19,
    land_mask: npt.NDArray[np.bool_],
    tb_mask: npt.NDArray[np.bool_],
    ln1,
    date: dt.date,
    weather_filter_seasons: list[WeatherFilterParamsForSeason],
) -> npt.NDArray[np.bool_]:
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
    watchk4 = fadd(fmul(ln1[1], v37), ln1[0])

    is_cond1 = (watchk1 > v19) | (watchk2 > wxlimt)
    # TODO: where does this 230.0 value come from? Should it be configuratble?
    is_cond2 = (watchk4 > h37) | (v37 >= 230.0)

    is_water = not_land_or_masked & is_cond1 & is_cond2

    return is_water


def calc_rad_coeffs_32(
    *,
    itp,
    wtp,
    vh37,
    itp2,
    wtp2,
    v1937,
):
    # Compute radlsp, radoff, radlen vars
    radslp1 = fdiv(
        fsub(f(itp[1]), f(wtp[1])),
        fsub(f(itp[0]), f(wtp[0])),
    )
    radoff1 = fsub(f(wtp[1]), fmul(f(wtp[0]), f(radslp1)))
    xint = fdiv(
        fsub(f(radoff1), f(vh37[0])),
        fsub(f(vh37[1]), f(radslp1)),
    )
    yint = fadd(fmul(vh37[1], f(xint)), f(vh37[0]))
    radlen1 = fsqt(
        fadd(
            fsqr(fsub(f(xint), f(wtp[0]))),
            fsqr(fsub(f(yint), f(wtp[1]))),
        )
    )

    radslp2 = fdiv(
        fsub(f(itp2[1]), f(wtp2[1])),
        fsub(f(itp2[0]), f(wtp2[0])),
    )
    radoff2 = fsub(f(wtp2[1]), fmul(f(wtp2[0]), f(radslp2)))
    xint = fdiv(
        fsub(f(radoff2), f(v1937[0])),
        fsub(f(v1937[1]), f(radslp2)),
    )
    yint = fadd(fmul(f(v1937[1]), f(xint)), f(v1937[0]))
    radlen2 = fsqt(
        fadd(
            fsqr(fsub(f(xint), f(wtp2[0]))),
            fsqr(fsub(f(yint), f(wtp2[1]))),
        )
    )

    return {
        'radslp1': radslp1,
        'radoff1': radoff1,
        'radlen1': radlen1,
        'radslp2': radslp2,
        'radoff2': radoff2,
        'radlen2': radlen2,
    }


def sst_clean_sb2(*, iceout, missval, landval, invalid_ice_mask: npt.NDArray[np.bool_]):
    # implement fortran's sst_clean_sb2() routine
    is_not_land = iceout != landval
    is_not_miss = iceout != missval
    is_not_land_miss_sst = is_not_land & is_not_miss & invalid_ice_mask

    ice_sst = iceout.copy()
    ice_sst[is_not_land_miss_sst] = 0.0

    return ice_sst


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


def calc_bt_ice(
    *,
    missval,
    landval,
    maxic,
    vh37,
    adoff,
    v1937,
    wtp,
    wtp2,
    itp,
    itp2,
    tb_v37: npt.NDArray,
    tb_h37: npt.NDArray,
    tb_v19: npt.NDArray,
    land_mask: npt.NDArray[np.bool_],
    water_mask: npt.NDArray[np.bool_],
    tb_mask: npt.NDArray[np.bool_],
):

    # ## LINES calculating radslp1 ... to radlen2 ###
    rad_coeffs = calc_rad_coeffs_32(
        itp=itp,
        wtp=wtp,
        vh37=vh37,
        itp2=itp2,
        wtp2=wtp2,
        v1937=v1937,
    )
    radslp1 = rad_coeffs['radslp1']
    radoff1 = rad_coeffs['radoff1']
    radlen1 = rad_coeffs['radlen1']
    radslp2 = rad_coeffs['radslp2']
    radoff2 = rad_coeffs['radoff2']
    radlen2 = rad_coeffs['radlen2']

    # main calc_bt_ice() block
    vh37chk = vh37[0] - adoff + vh37[1] * tb_v37

    # Compute radchk1
    is_check1 = tb_h37 > vh37chk
    is_h37_lt_rc1 = tb_h37 < (radslp1 * tb_v37 + radoff1)

    iclen1 = np.sqrt(np.square(tb_v37 - wtp[0]) + np.square(tb_h37 - wtp[1]))
    is_iclen1_gt_radlen1 = iclen1 > radlen1
    icpix1 = ret_ic_32(
        tb_v37,
        tb_h37,
        wtp[0],
        wtp[1],
        vh37[0],
        vh37[1],
        missval,
        maxic,
    )
    icpix1[is_h37_lt_rc1 & is_iclen1_gt_radlen1] = 1.0
    is_condition1 = is_h37_lt_rc1 & ~(iclen1 > radlen1)
    icpix1[is_condition1] = iclen1[is_condition1] / radlen1

    # Compute radchk2
    is_v19_lt_rc2 = tb_v19 < (radslp2 * tb_v37 + radoff2)

    iclen2 = np.sqrt(np.square(tb_v37 - wtp2[0]) + np.square(tb_v19 - wtp2[1]))
    is_iclen2_gt_radlen2 = iclen2 > radlen2
    icpix2 = ret_ic_32(
        tb_v37,
        tb_v19,
        wtp2[0],
        wtp2[1],
        v1937[0],
        v1937[1],
        missval,
        maxic,
    )
    icpix2[is_v19_lt_rc2 & is_iclen2_gt_radlen2] = 1.0
    is_condition2 = is_v19_lt_rc2 & ~is_iclen2_gt_radlen2
    icpix2[is_condition2] = iclen2[is_condition2] / radlen2

    ic = icpix1
    ic[~is_check1] = icpix2[~is_check1]

    is_ic_is_missval = ic == missval
    ic[is_ic_is_missval] = missval
    ic[~is_ic_is_missval] = ic[~is_ic_is_missval] * 100.0

    ic[water_mask] = 0.0
    ic[tb_mask] = 0.0
    ic[land_mask] = landval

    return ic


def get_hemisphere_from_gridid(gridid):
    if 'psn' in gridid or 'e2n' in gridid:
        return 'north'
    elif 'pss' in gridid or 'e2s' in gridid:
        return 'south'
    else:
        raise RuntimeError(f'Could not determine hemisphere from gridid {gridid}')


def get_intres_from_gridid(gridid):
    if '3.125' in gridid:
        return 3
    elif '6.25' in gridid:
        return 6
    elif '12.5' in gridid:
        return 12
    elif '25' in gridid:
        return 25
    else:
        raise RuntimeError(f'Could not determine hemisphere from gridid {gridid}')


def bootstrap_via_recipe(
    *,
    recipe: dict,
) -> xr.Dataset:

    # Initialize the dataset
    bt = xr.Dataset()

    # Start collecting algorithm parameters
    bt['icecon_parameters'] = int()
    bt['icecon_parameters'].attrs['icecon_algorithm'] = 'Bootstrap'

    # Parse parameters
    date = recipe['run_parameters']['date']
    hemisphere = get_hemisphere_from_gridid(recipe['run_parameters']['gridid'])
    intres = get_intres_from_gridid(recipe['run_parameters']['gridid'])

    bt['icecon_parameters'].attrs['gridid'] = recipe['run_parameters']['gridid']
    bt['icecon_parameters'].attrs['date'] = date

    # Read in the TBs
    # TODO: Will need to get 12.5km 6.9GHz fields here
    bt['icecon_parameters'].attrs['tb_source'] = 'get_au_si_tbs()'
    tbs = get_au_si_tbs(
        date=date,
        hemisphere=hemisphere,
        resolution=intres,
    )

    bt['tb_v37_in'] = tbs.variables['v36']
    bt['tb_h37_in'] = tbs.variables['h36']
    bt['tb_v19_in'] = tbs.variables['v18']
    bt['tb_v22_in'] = tbs.variables['v23']

    # Spatially interpolate the tb fields
    bt['icecon_parameters'].attrs['tb_spatial_interpolation'] = 'spatial_interp_tbs()'
    bt['tb_v37_si'] = (('y', 'x'), spatial_interp_tbs(bt['tb_v37_in'].data))
    bt['tb_h37_si'] = (('y', 'x'), spatial_interp_tbs(bt['tb_h37_in'].data))
    bt['tb_v19_si'] = (('y', 'x'), spatial_interp_tbs(bt['tb_v19_in'].data))
    bt['tb_v22_si'] = (('y', 'x'), spatial_interp_tbs(bt['tb_v22_in'].data))

    # Add the mask fields
    bt['icecon_parameters'].attrs['surface_mask'] = 'get_ps_land_mask()'
    surface_mask = get_ps_land_mask(hemisphere=hemisphere, resolution=str(intres))
    bt['surface_mask'] = (('y', 'x'), surface_mask)

    if hemisphere == 'north':
        pole_mask = get_ps_pole_hole_mask(resolution=str(intres))
        bt['pole_mask'] = (('y', 'x'), pole_mask)
        bt['icecon_parameters'].attrs['pole_mask'] = 'get_ps_pole_hole_mask()'
    else:
        bt['pole_mask'] = None

    # Note: invalid_ice_mask is of type:
    #  npt.NDArray[np.bool_]
    bt['icecon_parameters'].attrs['invalid_ice_mask'] = 'get_ps_invalid_ice_mask()'
    invalid_ice_mask = get_ps_invalid_ice_mask(
        hemisphere=hemisphere,
        date=date,
        resolution=str(intres),  # type: ignore[arg-type]
    )
    bt['invalid_ice_mask'] = (('y', 'x'), invalid_ice_mask)

    # Add sensor-specific BT parameters
    if hemisphere == 'north':
        bt['icecon_parameters'].attrs['bt_params_source'] = 'AMSR2_NORTH_PARAMS'
        for key in AMSR2_NORTH_PARAMS:
            bt['icecon_parameters'].attrs[key] = AMSR2_NORTH_PARAMS[key]

    else:
        bt['icecon_parameters'].attrs['bt_params_source'] = 'AMSR2_SOUTH_PARAMS'
        for key in AMSR2_SOUTH_PARAMS:
            bt['icecon_parameters'].attrs[key] = AMSR2_SOUTH_PARAMS[key]

    print(f'icecon_parameters var:\n{bt["icecon_parameters"]}')

    # NOTE: We should be able to construct the arguments to bootstrap() now

    return bt


def bootstrap(
    *,
    tb_v37: npt.NDArray,
    tb_h37: npt.NDArray,
    tb_v19: npt.NDArray,
    tb_v22: npt.NDArray,
    params: BootstrapParams,
    date: dt.date,
    hemisphere: Hemisphere,
    missing_flag_value: float | int = DEFAULT_FLAG_VALUES.missing,
) -> xr.Dataset:
    """Run the boostrap algorithm."""
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

    water_mask = ret_water_ssmi(
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

    vh37 = ret_linfit_32(
        land_mask=params.land_mask,
        tb_mask=tb_mask,
        tbx=tb_v37,
        tby=tb_h37,
        lnline=params.vh37_params.lnline,
        add=params.add1,
        water_mask=water_mask,
    )

    wtp, wtp2 = get_water_tiepoints(
        water_mask=water_mask,
        tb_v37=tb_v37,
        tb_h37=tb_h37,
        tb_v19=tb_v19,
        wtp1_default=params.vh37_params.water_tie_point,
        wtp2_default=params.v1937_params.water_tie_point,
    )

    adoff = ret_adj_adoff(wtp=wtp, vh37=vh37)

    # Try the ret_para... values for v1937
    v1937 = ret_linfit_32(
        land_mask=params.land_mask,
        tb_mask=tb_mask,
        tbx=tb_v37,
        tby=tb_v19,
        lnline=params.v1937_params.lnline,
        add=params.add2,
        water_mask=water_mask,
        tba=tb_h37,
        iceline=vh37,
        adoff=adoff,
    )

    # ## LINES with loop calling (in part) ret_ic() ###
    iceout = calc_bt_ice(
        missval=missing_flag_value,
        landval=DEFAULT_FLAG_VALUES.land,
        maxic=params.maxic,
        vh37=vh37,
        adoff=adoff,
        itp=params.vh37_params.ice_tie_point,
        itp2=params.v1937_params.ice_tie_point,
        wtp=wtp,
        wtp2=wtp2,
        v1937=v1937,
        tb_v37=tb_v37,
        tb_h37=tb_h37,
        tb_v19=tb_v19,
        land_mask=params.land_mask,
        water_mask=water_mask,
        tb_mask=tb_mask,
    )

    # *** Do sst cleaning ***
    print(f'before sst_clean, params:\n{params}')
    iceout_sst = sst_clean_sb2(
        iceout=iceout,
        missval=missing_flag_value,
        landval=DEFAULT_FLAG_VALUES.land,
        invalid_ice_mask=params.invalid_ice_mask,
    )

    # *** Do spatial interp ***
    iceout_fix = coastal_fix(
        iceout_sst, missing_flag_value, DEFAULT_FLAG_VALUES.land, params.minic
    )
    iceout_fix[iceout_fix < params.minic] = 0

    ds = xr.Dataset({'conc': (('y', 'x'), iceout_fix)})

    return ds
