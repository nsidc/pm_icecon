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
    TbSetParams,
    WeatherFilterParams,
    WeatherFilterParamsForSeason,
)
from pm_icecon.constants import DEFAULT_FLAG_VALUES
from pm_icecon.errors import BootstrapAlgError, UnexpectedSatelliteError

from pm_icecon.fetch.au_si import (
    AU_SI_RESOLUTIONS,
    get_au_si_tbs,
    get_au_si_tbs_zoomed,
)

from pm_icecon.bt.masks import get_ps_invalid_ice_mask
from pm_icecon.masks import (
    get_e2n625_land_mask,
    get_ps_land_mask,
    get_ps_pole_hole_mask,
)
from pm_icecon.interpolation import spatial_interp_tbs
from pm_icecon.bt.params.amsr2 import AMSR2_NORTH_PARAMS, AMSR2_SOUTH_PARAMS

# from pm_icecon.bt.api import amsr2_bootstrap
import pm_icecon.bt.compute_bt_ic as compute_bt_ic
"""
from pm_icecon.bt.compute_bt_ic import (
    tb_data_mask,
    ret_water_ssmi,
)
"""


def get_standard_bootstrap_recipe(gridid, tb_source, icecon_algorithm='bootstrap'):
    """Return a dictionary of the standard recipe for AU_SI12 bootstrap"""
    bt_recipe = {}

    bt_recipe['run_parameters'] = {
        'icecon_algorithm': icecon_algorithm,
        'gridid': gridid,
        'tb_source': 'au_si12',
        'date_str': '2020-01-01',
    }

    bt_recipe['tb_parameters'] = {
        'mintb': 10.0,
        'maxtb': 320.0,
    }

    if 'psn' in gridid and 'au_si' in tb_source:
        # These are the AMSR2_NORTH_PARAMS for NH AU_SI products
        bt_recipe['bootstrap_parameters'] = {
            # Old formulation...
            # 'vh37_params': {
            #     'water_tie_point': [207.2, 131.9],
            #     'ice_tie_point': [256.3, 241.2],
            #     'lnline': [-71.99, 1.20],
            # },
            # 'v1937_params': {
            #     'water_tie_point': [207.2, 182.4],
            #     'ice_tie_point': [256.3, 258.9],
            #     'lnline': [48.26, 0.8048],
            # },
            # weather_filter_seasons=[
            #     # November through April (`seas=1` in `boot_ice_amsru2_np.f`)
            #     WeatherFilterParamsForSeason(
            #         start_month=11,
            #         end_month=4,
            #         weather_filter_params=WeatherFilterParams(
            #             wintrc=84.73,
            #             wslope=0.5352,
            #             wxlimt=18.39,
            #         ),
            #     ),
            #     # May (`seas=2`) will get interpolated from the previous and next season
            #     # June through Sept. (`seas=3`)
            #     WeatherFilterParamsForSeason(
            #         start_month=6,
            #         end_month=9,
            #         weather_filter_params=WeatherFilterParams(
            #             wintrc=82.71,
            #             wslope=0.5352,
            #             wxlimt=23.34,
            #         ),
            #     ),
            #     # October (`seas=4`) will get interpolated from the previous and next
            #     # (first in this list) season.
            # ],

            # Current formulation...
            'wtp_v37_init': 207.2,
            'wtp_h37_init': 131.9,
            'wtp_v19_init': 182.4,

            'itp_v37': 256.3,
            'itp_h37': 241.2,
            'itp_v19': 258.9,

            'vh37_lnline_offset': -71.99,
            'vh37_lnline_slope': 1.20,

            'v1937_lnline_offset': 48.26,
            'v1937_lnline_slope': 0.8048,

            # These should be labeled 'wfseason_<n>'...
            # There will be at least one season (start_month=1, end_month=12)
            'wx_season_1_start_month': 11,
            'wx_season_1_end_month': 4,
            'wx_season_1_wintrc': 84.73,
            'wx_season_1_wslope': 0.5352,
            'wx_season_1_wxlimt': 18.39,

            'wx_season_2_start_month': 6,
            'wx_season_2_end_month': 9,
            'wx_season_2_wintrc': 82.71,
            'wx_season_2_wslope': 0.5352,
            'wx_season_2_wxlimt': 23.34,

            # These are used for AMSR2 7GHz weather filtering
            'wintrc2': 12.22,   # Northern Hemisphere
            'wslope2': 0.7020,  # Northern Hemisphere
            #'wintrc2': 10.93,   # Southern Hemisphere
            #'wslope2': 0.7046,  # Southern Hemisphere

            'add1': 0.0,
            'add2': -2.0,

            'minic': 10.0,
            'maxic': 1.0,

            'flag_value_missing': 255,
            'flag_value_land': 254,
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
        tb_v37 = (1.0170066 * tb_v37) + -4.9383355
        tb_h37 = (1.0009720 * tb_h37) + -1.3709822
        tb_v19 = (1.0140723 * tb_v19) + -3.4705583
        tb_v22 = (0.99652931 * tb_v22) + -0.82305684
    elif sat == 'f18':
        tb_v37 = (1.0104497 * tb_v37) + -3.3174017
        tb_h37 = (0.98914390 * tb_h37) + 1.2031835
        tb_v19 = (1.0057373 * tb_v19) + -0.92638520
        tb_v22 = (0.98793409 * tb_v22) + 1.2108198
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

    # Removing this fsqr() results in difference
    sumx2 = np.sum(fsqr(xvals), dtype=np.float64)
    # sumx2 = np.sum(np.square(xvals), dtype=np.float64)

    # sumy2 is included in Bootstrap, but not used.
    # sumy2 = np.sum(fsqr(yvals), dtype=np.float64)

    # Removing this fmul() causes difference
    sumxy = np.sum(fmul(xvals, yvals), dtype=np.float64)
    # sumxy = np.sum(xvals * yvals, dtype=np.float64)

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
        is_tba_le_modad = tba <= tbx * iceline[1] + iceline[0] - adoff
    else:
        is_tba_le_modad = np.full_like(not_land_or_masked, fill_value=True)

    is_tby_gt_lnline = tby > tbx * lnline[1] + lnline[0]

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

    fit_off = fadd(intrca, add)  # removing this fadd() causes difference!
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
    wintrc,
    wslope,
    wxlimt,
    v06=None,
    wintrc2=None,
    wslope2=None,
) -> npt.NDArray[np.bool_]:

    # Determine where there is definitely water
    not_land_or_masked = ~land_mask & ~tb_mask

    watchk1 = wslope * v22 + wintrc
    watchk2 = v22 - v19
    watchk4 = ln1[1] * v37 + ln1[0]

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
        f(itp[1]) - f(wtp[1]),
        f(itp[0]) - f(wtp[0]),
    )
    radoff1 = f(wtp[1]) - f(wtp[0]) * f(radslp1)
    xint = fdiv(
        f(radoff1) - f(vh37[0]),
        f(vh37[1]) - f(radslp1),
    )
    yint = vh37[1] * f(xint) + f(vh37[0])
    radlen1 = np.sqrt(
        (np.square(f(xint) - f(wtp[0])) + np.square(f(yint) - f(wtp[1])))
    )

    radslp2 = fdiv(
        f(itp2[1]) - f(wtp2[1]),
        f(itp2[0]) - f(wtp2[0]),
    )
    radoff2 = f(wtp2[1]) - f(wtp2[0]) * f(radslp2)
    xint = fdiv(
        f(radoff2) - f(v1937[0]),
        f(v1937[1]) - f(radslp2),
    )
    yint = f(v1937[1]) * f(xint) + f(v1937[0])
    radlen2 = np.sqrt(
        (np.square(f(xint) - f(wtp2[0])) + np.square(f(yint) - f(wtp2[1])))
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
    date = dt.datetime.strptime(recipe['run_parameters']['date_str'], '%Y-%m-%d').date()
    hemisphere = get_hemisphere_from_gridid(recipe['run_parameters']['gridid'])
    intres = get_intres_from_gridid(recipe['run_parameters']['gridid'])

    bt['icecon_parameters'].attrs['gridid'] = recipe['run_parameters']['gridid']
    bt['icecon_parameters'].attrs['date_string'] = date.strftime('%Y-%m-%d')

    # Add bootstrap-speciic parameters from recipe
    # Add sensor-specific BT parameters
    if hemisphere == 'north':
        bt['icecon_parameters'].attrs['bt_params_source'] = 'AMSR2_NORTH_PARAMS'
    else:
        bt['icecon_parameters'].attrs['bt_params_source'] = 'AMSR2_SOUTH_PARAMS'

    for bt_param in recipe['bootstrap_parameters']:
        bt['icecon_parameters'].attrs[bt_param] = recipe['bootstrap_parameters'][bt_param]

    # Read in the TBs
    # TODO: Will need to get 12.5km 6.9GHz fields here
    bt['icecon_parameters'].attrs['tb_source'] = 'get_au_si_tbs()'
    bt['icecon_parameters'].attrs['mintb'] = recipe['tb_parameters']['mintb']
    bt['icecon_parameters'].attrs['maxtb'] = recipe['tb_parameters']['maxtb']
    tbs = get_au_si_tbs(
        date=date,
        hemisphere=hemisphere,
        resolution=intres,
    )

    bt['tb_v37_in'] = tbs.variables['v36']
    bt['tb_h37_in'] = tbs.variables['h36']
    bt['tb_v19_in'] = tbs.variables['v18']
    bt['tb_v22_in'] = tbs.variables['v23']

    # The 6.9 GHz channels are only provided at 25km resolution,
    # so we "zoom" them to get them at other resolutions
    if intres == 25:
        bt['tb_v06_in'] = tbs.variables['v06']
    else:
        if intres == 12:
            zoom_factor = 2
        elif intres == 6:
            zoom_factor = 4
        else:
            raise RuntimeError('intres not recognized: {intres}')
        tbs_zoomed = get_au_si_tbs_zoomed(
            date=date,
            hemisphere=hemisphere,
            input_resolution='25',
            zoom_factor=zoom_factor,
            fields=['v06',]
        )

        bt['tb_v06_in'] = (('YDim', 'XDim'), tbs_zoomed['v06'])
        bt['tb_v06_in'].attrs['long_name'] = '6.9 GHz vertical daily average Tbs; bilinearly interpolated from 25km resolution'  # noqa
        bt['tb_v06_in'].attrs['units'] = 'degree_kelvin'
        bt['tb_v06_in'].attrs['standard_name'] = 'brightness_temperature'

    # Spatially interpolate the tb fields
    bt['icecon_parameters'].attrs['tb_spatial_interpolation'] = 'spatial_interp_tbs()'
    bt['tb_v37_si'] = (('y', 'x'), spatial_interp_tbs(bt['tb_v37_in'].data))
    bt['tb_h37_si'] = (('y', 'x'), spatial_interp_tbs(bt['tb_h37_in'].data))
    bt['tb_v19_si'] = (('y', 'x'), spatial_interp_tbs(bt['tb_v19_in'].data))
    bt['tb_v22_si'] = (('y', 'x'), spatial_interp_tbs(bt['tb_v22_in'].data))
    bt['tb_v06_si'] = (('y', 'x'), spatial_interp_tbs(bt['tb_v06_in'].data))

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

    a2bt_params = BootstrapParams(
        land_mask=surface_mask,
        pole_mask=pole_mask,
        invalid_ice_mask=invalid_ice_mask,
        vh37_params=TbSetParams(
            water_tie_point=[
                bt['icecon_parameters'].attrs['wtp_v37_init'],
                bt['icecon_parameters'].attrs['wtp_h37_init'],
            ],
            ice_tie_point=[
                bt['icecon_parameters'].attrs['itp_v37'],
                bt['icecon_parameters'].attrs['itp_h37'],
            ],
            lnline=[
                bt['icecon_parameters'].attrs['vh37_lnline_offset'],
                bt['icecon_parameters'].attrs['vh37_lnline_slope'],
            ],
        ),
        v1937_params=TbSetParams(
            water_tie_point=[
                bt['icecon_parameters'].attrs['wtp_v37_init'],
                bt['icecon_parameters'].attrs['wtp_v19_init'],
            ],
            ice_tie_point=[
                bt['icecon_parameters'].attrs['itp_v37'],
                bt['icecon_parameters'].attrs['itp_v19'],
            ],
            lnline=[
                bt['icecon_parameters'].attrs['v1937_lnline_offset'],
                bt['icecon_parameters'].attrs['v1937_lnline_slope'],
            ],
        ),
        weather_filter_seasons=[
            # November through April (`seas=1` in `boot_ice_amsru2_np.f`)
            WeatherFilterParamsForSeason(
                start_month=bt['icecon_parameters'].attrs['wx_season_1_start_month'],
                end_month=bt['icecon_parameters'].attrs['wx_season_1_end_month'],
                weather_filter_params=WeatherFilterParams(
                    wintrc=bt['icecon_parameters'].attrs['wx_season_1_wintrc'],
                    wslope=bt['icecon_parameters'].attrs['wx_season_1_wslope'],
                    wxlimt=bt['icecon_parameters'].attrs['wx_season_1_wxlimt'],
                ),
            ),
            # May (`seas=2`) will get interpolated from the previous and next season
            # June through Sept. (`seas=3`)
            WeatherFilterParamsForSeason(
                start_month=bt['icecon_parameters'].attrs['wx_season_2_start_month'],
                end_month=bt['icecon_parameters'].attrs['wx_season_2_end_month'],
                weather_filter_params=WeatherFilterParams(
                    wintrc=bt['icecon_parameters'].attrs['wx_season_2_wintrc'],
                    wslope=bt['icecon_parameters'].attrs['wx_season_2_wslope'],
                    wxlimt=bt['icecon_parameters'].attrs['wx_season_2_wxlimt'],
                ),
            ),
            # October (`seas=4`) will get interpolated from the previous and next
            # (first in this list) season.
        ],
    )
    a2bt_date = date
    a2bt_hemisphere = hemisphere

    # TODO: the weather filter parameters should be set when the parameters
    #       are read in.  The weather filter by season should not be a part
    #       of the bootstrap code.

    """
    # Call the bootstrap code via older method
    computed_bt_ds = compute_bt_ic.bootstrap(
        tb_v37=bt['tb_v37_si'].data,
        tb_h37=bt['tb_h37_si'].data,
        tb_v19=bt['tb_v19_si'].data,
        tb_v22=bt['tb_v22_si'].data,
        params=a2bt_params,
        date=a2bt_date,
        hemisphere=a2bt_hemisphere,
    )
    bt['icecon'] = computed_bt_ds['conc']
    """

    # Below, implement the functions of the compute_bt_ic.bootstrap() routine

    tb_data_mask_field = tb_data_mask(
        tbs=(
            bt['tb_v37_si'].data,
            bt['tb_h37_si'].data,
            bt['tb_v19_si'].data,
            bt['tb_v22_si'].data,
        ),
        min_tb=bt['icecon_parameters'].attrs['mintb'],
        max_tb=bt['icecon_parameters'].attrs['maxtb'],
    )

    bt['valid_tb_mask'] = (('y', 'x'), tb_data_mask_field)

    # Calcuate "normal" weather filter parameters
    season_params = _get_wx_params(
        date=date,
        weather_filter_seasons=a2bt_params.weather_filter_seasons,
    )
    bt['icecon_parameters'].attrs['wintrc'] = season_params.wintrc
    bt['icecon_parameters'].attrs['wslope'] = season_params.wslope
    bt['icecon_parameters'].attrs['wxlimt'] = season_params.wxlimt

    is_water_mask_field = ret_water_ssmi(
        v37=bt['tb_v37_si'].data,
        h37=bt['tb_h37_si'].data,
        v22=bt['tb_v22_si'].data,
        v19=bt['tb_v19_si'].data,
        land_mask=bt['surface_mask'].data,
        tb_mask=bt['valid_tb_mask'].data,
        ln1=[
            bt['icecon_parameters'].attrs['vh37_lnline_offset'],
            bt['icecon_parameters'].attrs['vh37_lnline_slope'],
        ],
        date=a2bt_date,
        weather_filter_seasons=a2bt_params.weather_filter_seasons,
        wintrc=bt['icecon_parameters'].attrs['wintrc'],
        wslope=bt['icecon_parameters'].attrs['wslope'],
        wxlimt=bt['icecon_parameters'].attrs['wxlimt'],
    )
    bt['is_water_mask'] =  (('y', 'x'), is_water_mask_field)

    # vh37 = ret_linfit_32(
    [bt['icecon_parameters'].attrs['vh37_linfitted_offset'],
     bt['icecon_parameters'].attrs['vh37_linfitted_slope']] = ret_linfit_32(
        land_mask=bt['surface_mask'],
        tb_mask=bt['valid_tb_mask'],
        tbx=bt['tb_v37_si'].data,
        tby=bt['tb_h37_si'].data,
        lnline=[
            bt['icecon_parameters'].attrs['vh37_lnline_offset'],
            bt['icecon_parameters'].attrs['vh37_lnline_slope'],
        ],
        add=bt['icecon_parameters'].attrs['add1'],
        water_mask=bt['is_water_mask'],
    )

    vh37_params_wtp = (
        bt['icecon_parameters'].attrs['wtp_v37_init'],
        bt['icecon_parameters'].attrs['wtp_h37_init'],
    )
    v1937_params_wtp = (
        bt['icecon_parameters'].attrs['wtp_v37_init'],
        bt['icecon_parameters'].attrs['wtp_v19_init'],
    )
    wtp, wtp2 = get_water_tiepoints(
        water_mask=bt['is_water_mask'].data,
        tb_v37=bt['tb_v37_si'].data,
        tb_h37=bt['tb_h37_si'].data,
        tb_v19=bt['tb_v19_si'].data,
        wtp1_default=vh37_params_wtp,
        wtp2_default=v1937_params_wtp,
    )

    (bt['icecon_parameters'].attrs['wtp_v37'],
     bt['icecon_parameters'].attrs['wtp_h37']) = wtp

    (bt['icecon_parameters'].attrs['wtp_v37'],
     bt['icecon_parameters'].attrs['wtp_v19']) = wtp2

    # adoff = ret_adj_adoff(
    bt['icecon_parameters'].attrs['adoff'] = ret_adj_adoff(
        wtp=(bt['icecon_parameters'].attrs['wtp_v37'], 
             bt['icecon_parameters'].attrs['wtp_h37']),
        vh37=[bt['icecon_parameters'].attrs['vh37_linfitted_offset'],
              bt['icecon_parameters'].attrs['vh37_linfitted_slope']],
    )

    v1937 = ret_linfit_32(
        land_mask=bt['surface_mask'],
        tb_mask=bt['valid_tb_mask'],
        tbx=bt['tb_v37_si'].data,
        tby=bt['tb_v19_si'].data,
        lnline=[
            bt['icecon_parameters'].attrs['v1937_lnline_offset'],
            bt['icecon_parameters'].attrs['v1937_lnline_slope'],
        ],
        add=bt['icecon_parameters'].attrs['add2'],
        water_mask=bt['is_water_mask'],
        tba=bt['tb_h37_si'].data,
        iceline=[bt['icecon_parameters'].attrs['vh37_linfitted_offset'],
                 bt['icecon_parameters'].attrs['vh37_linfitted_slope']],
        adoff=bt['icecon_parameters'].attrs['adoff'],
    )
    [bt['icecon_parameters'].attrs['v1937_linfitted_offset'],
     bt['icecon_parameters'].attrs['v1937_linfitted_slope']] = v1937

    iceout = calc_bt_ice(
        missval=bt['icecon_parameters'].attrs['flag_value_missing'],
        landval=bt['icecon_parameters'].attrs['flag_value_land'],
        maxic=bt['icecon_parameters'].attrs['maxic'],
        vh37=[bt['icecon_parameters'].attrs['vh37_linfitted_offset'],
              bt['icecon_parameters'].attrs['vh37_linfitted_slope']],
        adoff=bt['icecon_parameters'].attrs['adoff'],
        itp=[bt['icecon_parameters'].attrs['itp_v37'],
             bt['icecon_parameters'].attrs['itp_h37']],
        itp2=[bt['icecon_parameters'].attrs['itp_v37'],
              bt['icecon_parameters'].attrs['itp_v19']],
        wtp=(bt['icecon_parameters'].attrs['wtp_v37'],
             bt['icecon_parameters'].attrs['wtp_h37']),
        wtp2=(bt['icecon_parameters'].attrs['wtp_v37'],
              bt['icecon_parameters'].attrs['wtp_v19']),
        v1937=[bt['icecon_parameters'].attrs['v1937_linfitted_offset'],
               bt['icecon_parameters'].attrs['v1937_linfitted_slope']],
        tb_v37=bt['tb_v37_si'].data,
        tb_h37=bt['tb_h37_si'].data,
        tb_v19=bt['tb_v19_si'].data,
        land_mask=bt['surface_mask'],
        water_mask=bt['is_water_mask'],
        tb_mask=bt['valid_tb_mask'],
    )
    bt['iceout'] = (('y', 'x'), iceout)

    iceout_sst = sst_clean_sb2(
        iceout=bt['iceout'].data,
        missval=bt['icecon_parameters'].attrs['flag_value_missing'],
        landval=bt['icecon_parameters'].attrs['flag_value_land'],
        invalid_ice_mask=bt['invalid_ice_mask'].data,
    )
    bt['iceout_sst'] = (('y', 'x'), iceout_sst)

    iceout_fix = coastal_fix(
        bt['iceout_sst'].data,
        bt['icecon_parameters'].attrs['flag_value_missing'],
        bt['icecon_parameters'].attrs['flag_value_land'],
        bt['icecon_parameters'].attrs['minic'],
    )
    iceout_fix[iceout_fix < bt['icecon_parameters'].attrs['minic']] = 0

    bt['iceout_fix'] = (('y', 'x'), iceout_fix)
    bt['icecon'] = bt['iceout_fix']

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
