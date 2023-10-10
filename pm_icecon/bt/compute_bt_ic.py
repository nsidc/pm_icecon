"""Compute the Bootstrap ice concentration.

Takes values from part of boot_ice_sb2_ssmi_np_nrt.f
and computes:
    iceout
"""

import calendar
import copy
import datetime as dt
import warnings
from functools import reduce
from typing import Literal, Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr
from loguru import logger

from pm_icecon.bt._types import Line, Tiepoint, TiepointSet
from pm_icecon.config.models.bt import (
    BootstrapParams,
    WeatherFilterParams,
    WeatherFilterParamsForSeason,
)
from pm_icecon.constants import DEFAULT_FLAG_VALUES
from pm_icecon.errors import BootstrapAlgError, UnexpectedSatelliteError


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


def get_adj_ad_line_offset(
    *,
    wtp_x: Tiepoint,
    wtp_y: Tiepoint,
    # TODO: should this just be `line` instad of `line_37v37h`? Is this
    # function really specific to that set of channels? If so, maybe it's
    # worth changing `wtp_x` and `wpt_y` to more clearly indicate `wpt_37v`
    # and `wtp_37h`.
    line_37v37h: Line,
    perc=0.92,
) -> float:
    """Return the AD line offset.

    This version uses individual tie points, not a TiepointSet

    The AD line offset is used to determine between which tb set should be used
    to calculate a pixel's ice concentration. For the Goddard bootstrap
    algorithm, data points above the offset AD line (appox. AD - 5K) use
    HV37. For data points below the offset AD line, the V1937 tbs et is used
    instead.
    """
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

    ad_line_offset = off - new_off

    return ad_line_offset


def _get_wtp(
    weather_mask: npt.NDArray[np.bool_],
    tb: npt.NDArray[np.float32],
) -> Tiepoint:
    """Return the calculated water tiepoint (wtp) for the given Tbs."""
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
    thresh = nvals * pct
    while (ival < n_bins) and (subtotal < thresh):
        subtotal += histo[ival]
        ival += 1
    ival -= 1  # undo last increment

    wtp = Tiepoint(ival * 0.25)

    return wtp


def get_water_tiepoint_set(
    *,
    wtp_set_default: TiepointSet,
    weather_mask: npt.NDArray[np.bool_],
    tbx,
    tby,
) -> TiepointSet:
    """Return the deafult or calculate new water tiepoint set.

    If the calculated water tiepoints are within +/- 10 of the
    `wtp_set_default`, use the newly calculated values.
    """
    wtpx = _get_wtp(weather_mask, tbx)
    wtpy = _get_wtp(weather_mask, tby)

    new_wtp_set = list(copy.copy(wtp_set_default))

    # If the calculated wtps are within the bounds of the default (+/- 10), use
    # the calculated value.
    def _within_plusminus_10(initial_value, value) -> bool:
        return (initial_value - 10) < value < (initial_value + 10)

    if _within_plusminus_10(wtp_set_default[0], wtpx):
        new_wtp_set[0] = wtpx
    if _within_plusminus_10(wtp_set_default[1], wtpy):
        new_wtp_set[1] = wtpy

    wtp_set_tuple = (new_wtp_set[0], new_wtp_set[1])

    return wtp_set_tuple


def calculate_water_tiepoint(
    *,
    wtp_init: Tiepoint,
    weather_mask: npt.NDArray[np.bool_],
    tb,
) -> float:
    """Return the default or calculate new water tiepoint.

    If the calculated water tiepoints are within +/- 10 of the
    `wtp_set_default`, use the newly calculated values.
    """
    calculated_wtp = _get_wtp(weather_mask, tb)

    def _within_plusminus_10(initial_value, value) -> bool:
        return (initial_value - 10) < value < (initial_value + 10)

    if not _within_plusminus_10(wtp_init, calculated_wtp):
        calculated_wtp = Tiepoint(wtp_init)

    return calculated_wtp


def get_linfit(
    *,
    land_mask: npt.NDArray[np.bool_],
    tb_mask: npt.NDArray[np.bool_],
    tbx: npt.NDArray,
    tby: npt.NDArray,
    lnline: Line,
    add: float,
    weather_mask: npt.NDArray[np.bool_],
    # If the calculated slope is larger than `max_slope`, a `BootstrapAlgError`
    # is raised.
    max_slope: float = 1.5,
    # If any one of the rest of these arguments is given, the rest must also be
    # non-None. Currently only used for getting the v1937 line, in determining
    # if pixels are valid for use.
    tba: npt.NDArray | None = None,
    iceline: Line | None = None,
    ad_line_offset: float | None = None,
) -> Line:
    """Reproduce both `ret_linfit1()` and `ret_linfit2()` from GSFC code."""
    not_land_or_masked = ~land_mask & ~tb_mask
    # tba is always tb_h37, which is the x-axis of the 37h37v tbset.
    # The iceline is always the v19v37 lnline (ln == linear?).
    # ad_line_offset is calculated from the 37v37h tbset.
    if tba is not None and iceline is not None and ad_line_offset is not None:
        is_tba_le_modad = (
            tba <= (tbx * iceline['slope']) + iceline['offset'] - ad_line_offset
        )
    else:
        is_tba_le_modad = np.full_like(not_land_or_masked, fill_value=True)

    is_tby_gt_lnline = tby > (tbx * lnline['slope']) + lnline['offset']

    is_valid = not_land_or_masked & is_tba_le_modad & is_tby_gt_lnline & ~weather_mask

    num_valid_pixels = is_valid.sum()
    if num_valid_pixels <= 125:
        raise BootstrapAlgError(f'Insufficient valid linfit points: {num_valid_pixels}')

    slopeb, intrca = np.polyfit(
        x=tbx[is_valid],
        y=tby[is_valid],
        deg=1,
    )

    if slopeb > max_slope:
        raise BootstrapAlgError(
            f'Line slope check failed. {slopeb=} > {max_slope=}. '
            'This may need some additional investigation! The code from Goddard would'
            ' fall back on defaults defined by the `iceline` parameter if this'
            ' condition was met. However, it is probably better to investigate'
            ' this situation and determine what to do on a case-by-case basis'
            ' rather than "silently" fall back on some default values. We are not'
            ' sure how the default values of (`iceline`) were originally chosen.'
        )

    fit_off = intrca + add
    fit_slp = slopeb
    line = Line(offset=fit_off, slope=fit_slp)

    return line


def _get_ic(
    *,
    tbx: npt.NDArray,
    tby: npt.NDArray,
    wtp_xaxis: Tiepoint,
    wtp_yaxis: Tiepoint,
    iline: Line,
    missing_flag_value,
    maxic: float,
):
    """Get fractional ice concentration without rad adjustment."""
    wtp_x = wtp_xaxis
    wtp_y = wtp_yaxis
    iline_off = iline['offset']
    iline_slp = iline['slope']

    delta_x = tbx - wtp_x
    is_deltax_eq_0 = delta_x == 0

    # block1
    y_intercept = iline_off + iline_slp * tbx
    length1 = tby - wtp_y
    length2 = y_intercept - wtp_y
    ic_block1 = length1 / length2
    ic_block1[ic_block1 < 0] = 0
    ic_block1[ic_block1 > maxic] = maxic

    # block2
    delta_y = tby - wtp_y
    with warnings.catch_warnings():
        # This causes a divide-by-zero warning because
        # locations that are later ignored have zero in denominator
        warnings.simplefilter('ignore', category=RuntimeWarning)
        slope = delta_y / delta_x
    offset = tby - (slope * tbx)
    slp_diff = iline_slp - slope

    is_slp_diff_ne_0 = slp_diff != 0

    with warnings.catch_warnings():
        # This causes a divide-by-zero warning because
        # locations that are later ignored have zero in denominator
        warnings.simplefilter('ignore', category=RuntimeWarning)
        x_intercept = (offset - iline_off) / slp_diff
    y_intercept = offset + (slope * x_intercept)
    length1 = np.sqrt(np.square(tbx - wtp_x) + np.square(tby - wtp_y))
    length2 = np.sqrt(np.square(x_intercept - wtp_x) + np.square(y_intercept - wtp_y))
    ic_block2 = length1 / length2
    ic_block2[ic_block2 < 0] = 0
    ic_block2[ic_block2 > maxic] = maxic
    ic_block2[~is_slp_diff_ne_0] = missing_flag_value

    # Assume ic is block2, then overwrite if block1
    ic = ic_block2
    ic[is_deltax_eq_0] = ic_block1[is_deltax_eq_0]

    return ic


def _get_len_between_points(
    *,
    x1: npt.NDArray | float,
    y1: npt.NDArray | float,
    x2: npt.NDArray | float,
    y2: npt.NDArray | float,
) -> npt.NDArray:
    """Return the length between (x1, y1) and (x2, y2).

    In practice, this is used for finding the distance between a tiepoint set
    and a tbset. E.g., the distance between 37v19v and the water tiepoint set
    for 37v19v.
    """
    length = np.sqrt(np.square(x1 - x2) + np.square(y1 - y2))

    return length


def calc_rad_coeffs(
    *,
    wtp_xaxis: Tiepoint,
    wtp_yaxis: Tiepoint,
    itp_xaxis: Tiepoint,
    itp_yaxis: Tiepoint,
    line: Line,
):
    rad_slope = (itp_yaxis - wtp_yaxis) / (itp_xaxis - wtp_xaxis)
    rad_offset = wtp_yaxis - (wtp_xaxis * rad_slope)

    xint = (rad_offset - line['offset']) / (line['slope'] - rad_slope)
    yint = (line['slope'] * xint) + line['offset']

    rad_len = _get_len_between_points(x1=xint, x2=wtp_xaxis, y1=yint, y2=wtp_yaxis)

    return (rad_slope, rad_offset, rad_len)


def _rad_adjust_ic(
    *,
    ic: npt.NDArray,
    tbx: npt.NDArray,
    tby: npt.NDArray,
    wtp_xaxis: Tiepoint,
    wtp_yaxis: Tiepoint,
    itp_xaxis: Tiepoint,
    itp_yaxis: Tiepoint,
    line: Line,
):
    adjusted_ic = ic.copy()

    radslp, rad_line_offset, radlen = calc_rad_coeffs(
        wtp_xaxis=wtp_xaxis,
        wtp_yaxis=wtp_yaxis,
        itp_xaxis=itp_xaxis,
        itp_yaxis=itp_yaxis,
        line=line,
    )

    is_tby_lt_rc = tby < (radslp * tbx + rad_line_offset)
    iclen = _get_len_between_points(x1=tbx, x2=wtp_xaxis, y1=tby, y2=wtp_yaxis)
    is_iclen_gt_radlen = iclen > radlen
    adjusted_ic[is_tby_lt_rc & is_iclen_gt_radlen] = 1.0
    is_condition = is_tby_lt_rc & ~is_iclen_gt_radlen
    adjusted_ic[is_condition] = iclen[is_condition] / radlen

    return adjusted_ic


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
    wintrc,
    wslope,
    wxlimt,
) -> npt.NDArray[np.bool_]:
    """Return a water mask that has been weather filtered.

    `True` indicates areas that are water and are weather masked. I.e., `True`
    values should be treated as open ocean.
    """
    # Determine where there is definitely water
    not_land_or_masked = ~land_mask & ~tb_mask
    watchk1 = (wslope * v22) + wintrc
    watchk2 = v22 - v19
    watchk4 = (ln1['slope'] * v37) + ln1['offset']

    is_cond1 = (watchk1 > v19) | (watchk2 > wxlimt)
    # TODO: where does this 230.0 value come from? Should it be configuratble?
    is_cond2 = (watchk4 > h37) | (v37 >= 230.0)

    is_water = not_land_or_masked & is_cond1 & is_cond2

    return is_water


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
    is_not_land_sst = is_not_land & invalid_ice_mask

    ice_sst = conc.copy()
    ice_sst[is_not_land_sst] = 0.0

    return ice_sst


def coastal_fix(
    *,
    conc: npt.NDArray,
    missing_flag_value,
    land_mask: npt.NDArray[np.bool_],
    # The minimum ice concentration as a percentage (10 == 10%)
    minic: float,
):
    # Apply coastal_fix() routine per Bootstrap.

    # Calculate 'temp' array
    #   -1 is no ice
    #    1 is safe from removal
    #    0 is might-be-removed
    temp = np.ones_like(conc, dtype=np.int16)
    is_land_or_lowice = land_mask | ((conc >= 0) & (conc < minic))
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
        is_rolled_land = np.roll(land_mask, offp1, axis=(1, 0))  # land
        rolled_offp1 = np.roll(conc, offn1, axis=(1, 0))  # k1 k2p1
        rolled_offp1_land_mask = np.roll(land_mask, offn1, axis=(1, 0))  # k1 k2p1
        rolled_offp2 = np.roll(conc, offn2, axis=(1, 0))  # k2

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

        is_k1p0 = (is_k1) & (conc > 0) & (conc != missing_flag_value) & (~land_mask)
        is_k2p0 = (is_k2) & (conc > 0) & (conc != missing_flag_value) & (~land_mask)
        is_k2p1 = (
            (is_k2)
            & (rolled_offp1 > 0)
            & (rolled_offp1 != missing_flag_value)
            & (~rolled_offp1_land_mask)
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

    def _conc_change124(*, shifts, land_mask, temp, conc2):
        # Compute shifted conc grid, for land check
        is_rolled_land = np.roll(land_mask, shifts[0], axis=(1, 0))

        is_temp0 = temp == 0
        is_considered = is_temp0 & is_rolled_land

        # For offp1 of [0, 1], the rolls are:
        tip1jp1 = np.roll(temp, shifts[1], axis=(1, 0))
        tim1jp1 = np.roll(temp, shifts[2], axis=(1, 0))
        tip1jp0 = np.roll(temp, shifts[3], axis=(1, 0))
        tim1jp0 = np.roll(temp, shifts[4], axis=(1, 0))
        tip0jp1 = np.roll(temp, shifts[5], axis=(1, 0))

        is_tip1jp1_lt0 = tip1jp1 <= 0
        is_tim1jp1_lt0 = tim1jp1 <= 0
        is_tip1jp0_lt0 = tip1jp0 <= 0
        is_tim1jp0_lt0 = tim1jp0 <= 0
        is_tip0jp1_eq0 = tip0jp1 == 0

        # Changing conc2(i,j+1) to 0
        locs_ip0jp1 = np.where(
            is_considered & is_tip1jp1_lt0 & is_tim1jp1_lt0 & is_tip0jp1_eq0
        )
        change_locs_conc2_ip0jp1 = tuple(
            [locs_ip0jp1[0] + shifts[0][1], locs_ip0jp1[1] + shifts[0][0]]
        )
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

        return conc2

    conc2 = _conc_change124(
        shifts=[
            (0, 1),
            (-1, -1),
            (1, -1),
            (-1, 0),
            (1, 0),
            (0, -1),
        ],
        land_mask=land_mask,
        temp=temp,
        conc2=conc2,
    )

    # Second conc2 change section
    conc2 = _conc_change124(
        shifts=[
            (0, -1),
            (-1, 1),
            (1, 1),
            (-1, 0),
            (1, 0),
            (0, 1),
        ],
        land_mask=land_mask,
        temp=temp,
        conc2=conc2,
    )

    # Third conc2 change section

    # Compute shifted conc grid, for land check
    is_rolled_land = np.roll(land_mask, (1, 0), axis=(1, 0))

    is_temp0 = temp == 0
    is_considered = is_temp0 & is_rolled_land

    # args to np.roll are opposite of fortran index offsets
    # TODO: one less roll operation here than in `_conc_change124`. Should there
    # be?
    tip1jp1 = np.roll(temp, (-1, -1), axis=(1, 0))
    tip1jp0 = np.roll(temp, (-1, 0), axis=(1, 0))
    tip0jm1 = np.roll(temp, (0, 1), axis=(1, 0))
    tip0jp1 = np.roll(temp, (0, -1), axis=(1, 0))

    # TODO: one less conditional here than in `_conc_change124`. Should there
    # be?
    is_tip1jp1_le0 = tip1jp1 <= 0
    is_tip1jp0_eq0 = tip1jp0 == 0
    is_tip0jm1_le0 = tip0jm1 <= 0
    is_tip0jp1_le0 = tip0jp1 <= 0

    # Changing conc2(i+1,j) to 0
    # TODO: `is_tip1jp1_le0` variable is used twice here (&-ed
    # together). Was this a mistake?
    locs_ip1jp0 = np.where(
        is_considered & is_tip1jp1_le0 & is_tip1jp1_le0 & is_tip1jp0_eq0
    )
    change_locs_conc2_ip1jp0 = tuple([locs_ip1jp0[0] + 0, locs_ip1jp0[1] + 1])
    conc2[change_locs_conc2_ip1jp0] = 0

    # Changing conc2(i,j) to 0
    # TODO: `is_tip1jp1_le0` variable is used twice here (&-ed
    # together). Was this a mistake?
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
    conc2 = _conc_change124(
        shifts=[
            (-1, 0),
            (1, 1),
            (1, -1),
            (0, 1),
            (0, -1),
            (1, 0),
        ],
        land_mask=land_mask,
        temp=temp,
        conc2=conc2,
    )

    return conc2


def _calc_frac_conc_for_tbset(
    *,
    tbx,
    tby,
    wtp_xaxis: Tiepoint,
    wtp_yaxis: Tiepoint,
    itp_xaxis: Tiepoint,
    itp_yaxis: Tiepoint,
    line: Line,
    missing_flag_value: float | int,
    maxic,
):
    """Return fractional sea ice concentration for the given parameters."""
    ic = _get_ic(
        tbx=tbx,
        tby=tby,
        wtp_xaxis=wtp_xaxis,
        wtp_yaxis=wtp_yaxis,
        iline=line,
        missing_flag_value=missing_flag_value,
        maxic=maxic,
    )

    ic_adjusted = _rad_adjust_ic(
        ic=ic,
        tbx=tbx,
        tby=tby,
        wtp_xaxis=wtp_xaxis,
        wtp_yaxis=wtp_yaxis,
        itp_xaxis=itp_xaxis,
        itp_yaxis=itp_yaxis,
        line=line,
    )

    return ic_adjusted


def calc_bootstrap_conc(
    *,
    tb_v37: npt.NDArray,
    tb_h37: npt.NDArray,
    tb_v19: npt.NDArray,
    wtp_37v: Tiepoint,
    wtp_37h: Tiepoint,
    wtp_19v: Tiepoint,
    itp_37v: Tiepoint,
    itp_37h: Tiepoint,
    itp_19v: Tiepoint,
    line_37v37h: Line,
    line_37v19v: Line,
    ad_line_offset: float,
    maxic_frac,
    missing_flag_value: float | int,
):
    """Return a sea ice concentration estimate at every grid cell.

    Concentrations are given as percentage (0-100%).
    """
    ic_frac_37v37h = _calc_frac_conc_for_tbset(
        tbx=tb_v37,
        tby=tb_h37,
        wtp_xaxis=wtp_37v,
        wtp_yaxis=wtp_37h,
        itp_xaxis=itp_37v,
        itp_yaxis=itp_37h,
        line=line_37v37h,
        missing_flag_value=missing_flag_value,
        maxic=maxic_frac,
    )

    ic_frac_37v19v = _calc_frac_conc_for_tbset(
        tbx=tb_v37,
        tby=tb_v19,
        wtp_xaxis=wtp_37v,
        wtp_yaxis=wtp_19v,
        itp_xaxis=itp_37v,
        itp_yaxis=itp_19v,
        line=line_37v19v,
        missing_flag_value=missing_flag_value,
        maxic=maxic_frac,
    )

    # Initialize the ice fraction from the 37v37h tbset. These values will be
    # preserved for pixels where tb_h37 is above the 37v37h AD line.
    ic_frac = ic_frac_37v37h.copy()
    # Use conc from the 37v19v tbset when tb_h37 is below the 37v37h AD line.
    ad_line_37v37h_y_vals = (
        line_37v37h['offset'] - ad_line_offset + line_37v37h['slope'] * tb_v37
    )
    h37_below_37v37h_ad_line = tb_h37 <= ad_line_37v37h_y_vals
    ic_frac[h37_below_37v37h_ad_line] = ic_frac_37v19v[h37_below_37v37h_ad_line]

    # convert fractional sea ice concentrations to percentages
    ic_perc = ic_frac.copy()
    # TODO/NOTE: if we treat missing as `np.nan`, this comparison does not
    # work. `np.nan != np.nan`.
    is_missing = ic_frac == missing_flag_value
    ic_perc[~is_missing] = ic_frac[~is_missing] * 100.0

    return ic_perc


def bootstrap_for_cdr(
    tb_v37: npt.NDArray,
    tb_h37: npt.NDArray,
    tb_v19: npt.NDArray,
    params: BootstrapParams,
    tb_mask: npt.NDArray[np.bool_],
    weather_mask: npt.NDArray[np.bool_],
    missing_flag_value: float | int = DEFAULT_FLAG_VALUES.missing,
    dont_use_tiepointset: bool = False,
) -> npt.NDArray:
    """Calculate raw Bootstrap sea ice concentration field.

    Returns an NDArray with a concentration estimate at every cell.

    Flags values are not set and spillover correction is not applied.
    """
    line_37v37h = get_linfit(
        land_mask=params.land_mask,
        tb_mask=tb_mask,
        tbx=tb_v37,
        tby=tb_h37,
        lnline=params.vh37_params.lnline,
        add=params.add1,
        weather_mask=weather_mask,
    )

    if dont_use_tiepointset:
        # This is a dummy set of function calls because vulture
        # declares these methods as unused...but they will eventually
        # be used to replace the current methods that use TiepointSet
        # variables...which we are trying to get away from

        # calculate_water_tiepoint() will be called for each water tiepoint
        wtp_tb_v37 = calculate_water_tiepoint(
            wtp_init=Tiepoint(12.0),  # this "12" is just a dummy placeholder
            weather_mask=weather_mask,
            tb=tb_v37,
        )
        assert wtp_tb_v37 is not None

        ad_line_offset = get_adj_ad_line_offset(
            wtp_x=Tiepoint(12.3),
            wtp_y=Tiepoint(24.8),
            line_37v37h=line_37v37h,
            perc=0.92,
        )

        weather_mask = get_weather_mask(
            v37=tb_v37,
            h37=tb_h37,
            v22=tb_v37,
            v19=tb_v19,
            land_mask=params.land_mask,
            tb_mask=tb_mask,
            ln1=params.vh37_params.lnline,
            date=dt.date(2000, 1, 1),
            wintrc=1.2,  # dummy placeholder var
            wslope=2.3,  # dummy placeholder var
            wxlimt=3.4,  # dummy placeholder var
        )

    wtp_set_37v37h = get_water_tiepoint_set(
        wtp_set_default=params.vh37_params.water_tie_point_set,
        weather_mask=weather_mask,
        tbx=tb_v37,
        tby=tb_h37,
    )

    wtp_set_37v19v = get_water_tiepoint_set(
        wtp_set_default=params.v1937_params.water_tie_point_set,
        weather_mask=weather_mask,
        tbx=tb_v37,
        tby=tb_v19,
    )

    ad_line_offset = get_adj_ad_line_offset(
        wtp_x=wtp_set_37v37h[0],
        wtp_y=wtp_set_37v37h[1],
        line_37v37h=line_37v37h,
    )

    line_37v19v = get_linfit(
        land_mask=params.land_mask,
        tb_mask=tb_mask,
        tbx=tb_v37,
        tby=tb_v19,
        lnline=params.v1937_params.lnline,
        add=params.add2,
        weather_mask=weather_mask,
        tba=tb_h37,
        iceline=line_37v37h,
        ad_line_offset=ad_line_offset,
    )

    conc = calc_bootstrap_conc(
        tb_v37=tb_v37,
        tb_h37=tb_h37,
        tb_v19=tb_v19,
        wtp_37v=wtp_set_37v37h[0],
        wtp_37h=wtp_set_37v37h[1],
        wtp_19v=wtp_set_37v19v[1],
        itp_37v=params.vh37_params.ice_tie_point_set[0],
        itp_37h=params.vh37_params.ice_tie_point_set[1],
        itp_19v=params.v1937_params.ice_tie_point_set[1],
        line_37v37h=line_37v37h,
        line_37v19v=line_37v19v,
        ad_line_offset=ad_line_offset,
        maxic_frac=params.maxic,
        missing_flag_value=missing_flag_value,
    )

    """ Original ordering of call to calc_bootstrap_conc()
    conc = calc_bootstrap_conc(
        maxic_frac=params.maxic,
        line_37v37h=line_37v37h,
        ad_line_offset=ad_line_offset,
        line_37v19v=line_37v19v,
        wtp_set_37v37h=wtp_set_37v37h,
        wtp_set_37v19v=wtp_set_37v19v,
        itp_set_37v37h=params.vh37_params.ice_tie_point_set,
        itp_set_37v19v=params.v1937_params.ice_tie_point_set,
        tb_v37=tb_v37,
        tb_h37=tb_h37,
        tb_v19=tb_v19,
        missing_flag_value=missing_flag_value,
    )
    """

    return conc


# TODO: This pole hole logic should be refactored.
#       Specifically, the definition of the pixels for which missing data
#       will be considered "pole hole" rather than simply "missing (because
#       of lack of sensor observation)" is on the same level of abstraction
#       as a "land_mask", and therefore should be identified and stored as
#       ancillary data in a similar location and with similar level of
#       description, including the derivation of the set of grid cells
#       identified as "pole hole".
def fill_pole_hole_bt(conc):
    """Fill the pole hole with the average of nearby missing values.

    TODO: This routine needs a better way of determining how big the pole
    hole region should be rather than assumptions based on grid size.
    """
    ydim, xdim = conc.shape

    # TODO: This logic can be tightened up.  Multiples of 720 indicate
    #       that we are using an EASE2 grid, and that the North Pole --
    #       and therefore the pole hole -- is near the center of the grid.
    #  For the polar stereo grid (see below) the pole hole pixels are
    #       specified by manually creating a pole hole mask kernel.
    pole_radius = 50
    grid_projection = 'EASE2'
    if xdim == 3360:
        pole_radius = 30
    elif xdim == 1680:
        pole_radius = 15
    elif xdim == 840:
        pole_radius = 8
    elif xdim == 720:
        pole_radius = 10
    elif xdim == 304:
        grid_projection = 'PS'
    elif xdim == 304:
        grid_projection = 'PS'
    else:
        raise ValueError(f'Could not determine pole_radius for xdim: {xdim}')

    if grid_projection == 'EASE2':
        half_ydim = ydim // 2
        half_xdim = xdim // 2

        # Note: near_pole_conc is a view into the pole-hole region of conc
        near_pole_conc = conc[
            half_ydim - pole_radius : half_ydim + pole_radius,
            half_xdim - pole_radius : half_xdim + pole_radius,
        ]
    elif grid_projection == 'PS':
        ph25ymin = 230
        ph25ymax = 238
        ph25xmin = 150
        ph25xmax = 157

        if xdim == 304:
            # Use pixel set appropriate for AMSR2 pole hole on 25km PSN grid
            near_pole_conc = conc[ph25ymin:ph25ymax, ph25xmin:ph25xmax]
        elif xdim == 608:
            # Use pixel set appropriate for AMSR2 pole hole on 12.5km PSN grid
            near_pole_conc = conc[
                ph25ymin * 2 : ph25ymax * 2, ph25xmin * 2 : ph25xmax * 2
            ]
        elif xdim == 1216:
            # Use pixel set appropriate for AMSR2 pole hole on 6.25km PSN grid
            near_pole_conc = conc[
                ph25ymin * 4 : ph25ymax * 4, ph25xmin * 4 : ph25xmax * 4
            ]
        else:
            raise ValueError(
                f'Expecting NH polar stereo, but unrecognized xdim: {xdim}'
            )

    is_pole_hole = (near_pole_conc < 0.01) | (near_pole_conc > 100)

    near_pole_mean = np.mean(near_pole_conc[~is_pole_hole])

    near_pole_conc[is_pole_hole] = near_pole_mean

    logger.info(f'Filled missing values at pole hole with: {near_pole_mean}')

    return conc


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

    season_params = _get_wx_params(
        date=date,
        weather_filter_seasons=params.weather_filter_seasons,
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
        wintrc=season_params.wintrc,
        wslope=season_params.wslope,
        wxlimt=season_params.wxlimt,
    )

    conc = bootstrap_for_cdr(
        tb_v37=tb_v37,
        tb_h37=tb_h37,
        tb_v19=tb_v19,
        params=params,
        tb_mask=tb_mask,
        weather_mask=weather_mask,
        missing_flag_value=missing_flag_value,
    )

    # Apply masks and flag values
    conc[weather_mask] = 0.0
    conc[tb_mask] = DEFAULT_FLAG_VALUES.missing
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
        land_mask=params.land_mask,
        minic=params.minic,
    )
    conc[conc < params.minic] = 0

    jdim, idim = conc.shape
    # If middle of land_mask is land, this is SH and needs no pole hole fill
    if not params.land_mask[jdim // 2, idim // 2]:
        conc = fill_pole_hole_bt(conc)

    ds = xr.Dataset({'conc': (('y', 'x'), conc)})

    return ds
