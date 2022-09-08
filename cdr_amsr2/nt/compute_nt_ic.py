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

import datetime as dt
from typing import Any

import numpy as np
import numpy.typing as npt
import xarray as xr
from loguru import logger

from cdr_amsr2._types import Hemisphere, ValidSatellites
from cdr_amsr2.constants import DEFAULT_FLAG_VALUES
from cdr_amsr2.nt.tiepoints import get_tiepoints


def fdiv(a, b):
    return np.divide(a, b, dtype=np.float32)


def nt_spatint(tbs):
    # Implement spatial interpolation scheme of SpatialInt_np.c
    # and SpatialInt_sp.c
    # Weighting scheme is: orthogonally adjacent weighted 1.0
    #                      diagonally adjacent weighted 0.707
    interp_tbs = {}
    for tb in tbs.keys():
        orig = tbs[tb].copy()
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

        interp = orig.copy()
        interp[replace_locs] = average[replace_locs]

        interp_tbs[tb] = interp

    return interp_tbs


def compute_nt_coefficients(tp: dict[str, dict[str, float]]) -> dict[str, float]:
    """Compute coefficients for the NT algorithm.

    tp are the tiepoints, a dictionary of structure:
      tp[channel][tiepoint]
         where channel is 'v19', 'h19', 'v37'
               tiepoint is 'ow', 'my', 'fy'
                        for open water, multiyear, first-year respectively
    """
    # Intermediate variables
    # TODO: better type annotations.
    diff: dict[str, dict[str, Any]] = {}
    sums: dict[str, dict[str, Any]] = {}
    for tiepoint in ('ow', 'fy', 'my'):
        diff[tiepoint] = {}
        diff[tiepoint]['19v19h'] = tp['19v'][tiepoint] - tp['19h'][tiepoint]
        diff[tiepoint]['37v19v'] = tp['37v'][tiepoint] - tp['19v'][tiepoint]

        sums[tiepoint] = {}
        sums[tiepoint]['19v19h'] = tp['19v'][tiepoint] + tp['19h'][tiepoint]
        sums[tiepoint]['37v19v'] = tp['37v'][tiepoint] + tp['19v'][tiepoint]

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


def compute_ratios(tbs, coefs) -> dict[str, npt.NDArray]:
    """Return calculated gradient ratios.

    TODO: make this function more generic. There should be a func for computing
    a single gradient ratio. This function could call that.
    """
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


def get_gr_thresholds(sat: ValidSatellites, hem: Hemisphere) -> dict[str, float]:
    """Return the gradient ratio thresholds for this sat, hem combo."""
    gr_thresholds = {}
    if sat == '17_final' or sat == 'u2':
        if sat == 'u2':
            logger.warning(
                'The graident threshold values were stolen from f17_final!'
                ' Do we need new ones for AMSR2? How do we get them?'
            )
        if hem == 'north':
            gr_thresholds['3719'] = 0.050
            gr_thresholds['2219'] = 0.045
        else:
            gr_thresholds['3719'] = 0.053
            gr_thresholds['2219'] = 0.045

    return gr_thresholds


def apply_weather_filter(
    conc: npt.NDArray, ratios: dict[str, npt.NDArray], thres: dict[str, float]
) -> npt.NDArray:
    """Return a copy of `conc` masked by the weather filter."""
    # Determine where array is weather-filtered
    print(f'thres 2219: {thres["2219"]}')
    print(f'thres 3719: {thres["3719"]}')

    filtered = (ratios['gr_2219'] > thres['2219']) | (ratios['gr_3719'] > thres['3719'])

    filtered_conc = conc.copy()
    filtered_conc[filtered] = 0

    return filtered_conc


def apply_invalid_tbs_mask(
    conc: npt.NDArray, tbs: dict[str, npt.NDArray]
) -> npt.NDArray:
    """Return a copy of `conc` masked where TBs are invalid.

    Invalid elements are set to a value of -10 in the concentration field.
    """
    is_valid_tbs = (tbs['v19'] > 0) & (tbs['h19'] > 0) & (tbs['v37'] > 0)

    filtered_conc = conc.copy()
    # TODO: would it be better to have a flag value specifically for invalid
    # data? Is treating this as 'missing' really appropriate?
    filtered_conc[~is_valid_tbs] = DEFAULT_FLAG_VALUES.missing

    return filtered_conc


def compute_nt_conc(
    tbs: dict[str, npt.NDArray], coefs: dict[str, float], ratios: dict[str, npt.NDArray]
) -> npt.NDArray:
    """Compute NASA Team sea ice concentration estimate."""
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


def apply_nt_spillover(
    *, conc_int16: npt.NDArray[np.int16], shoremap: npt.NDArray, minic: npt.NDArray
) -> npt.NDArray[np.int16]:
    """Apply the NASA Team land spillover routine."""
    newice = conc_int16.copy()

    # Set land/coast flag values.
    # TODO: do we want the coast to be 'land' as it is in bootstrap?
    # 1 == land
    newice[shoremap == 1] = DEFAULT_FLAG_VALUES.land
    # 2 == coast
    # TODO: re-add this flag. For now, making the flags for nt consistent w/ bt.
    # newice[shoremap == 2] = DEFAULT_FLAG_VALUES.coast
    newice[shoremap == 2] = DEFAULT_FLAG_VALUES.land

    is_at_coast = shoremap == 5
    is_near_coast = shoremap == 4
    is_far_coastal = shoremap == 3

    minic[is_at_coast & (minic > 200)] = 200
    minic[is_near_coast & (minic > 400)] = 400
    minic[is_far_coastal & (minic > 600)] = 600

    # Count number of nearby low ice conc
    n_low = np.zeros_like(conc_int16, dtype=np.uint8)

    for joff in range(-3, 3 + 1):
        for ioff in range(-3, 3 + 1):
            offmax = max(abs(ioff), abs(joff))

            rolled = np.roll(conc_int16, (joff, ioff), axis=(0, 1))
            is_rolled_low = (rolled < 150) & (rolled >= 0)

            if offmax <= 1:
                n_low[is_rolled_low & is_at_coast] += 1

            if offmax <= 2:
                n_low[is_rolled_low & is_near_coast] += 1

            if offmax <= 3:
                n_low[is_rolled_low & is_far_coastal] += 1

    # Note: there are meaningless differences "at the edge" in these counts
    # because the spatial interpolation is not identical along the border

    where_reduce_ice = (n_low >= 3) & (shoremap > 2)
    newice[where_reduce_ice] -= minic[where_reduce_ice]

    where_ice_overreduced = (conc_int16 >= 0) & (newice < 0) & (shoremap > 2)
    newice[where_ice_overreduced] = 0

    # Preserve missing data (conc value of -10)
    where_missing = (conc_int16 < 0) & where_reduce_ice & (shoremap > 2)
    newice[where_missing] = conc_int16[where_missing]

    return newice


def apply_invalid_icemask(
    *, conc: npt.NDArray[np.int16], invalid_ice_mask: npt.NDArray[np.bool_]
) -> npt.NDArray[np.int16]:
    """Replace all `True` elements in the invalid ice mask with 0."""
    masked_conc = np.where(invalid_ice_mask, 0, conc.copy())

    return masked_conc


def nasateam(
    *,
    tbs: dict[str, npt.NDArray],
    sat: ValidSatellites,
    hemisphere: Hemisphere,
    shoremap: npt.NDArray,
    minic: npt.NDArray,
    date: dt.date,
    invalid_ice_mask: npt.NDArray[np.bool_],
):
    spi_tbs = nt_spatint(tbs)

    # Here, the tbs are identical to the output of the Goddard code

    # TODO: is this already implemented? Do we need this comment?
    # TODO: what is this 'next step to implement'?
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
    tiepoints = get_tiepoints(satellite=sat, hemisphere=hemisphere)
    print(f'tiepoints: {tiepoints}')

    nt_coefficients = compute_nt_coefficients(tiepoints)
    print(f'NT coefs: {nt_coefficients}')
    for c in ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L'):
        print(f'  coef {c}: {nt_coefficients[c]}')

    gr_thresholds = get_gr_thresholds(sat, hemisphere)
    print(f'gr_thresholds:\n{gr_thresholds}')

    ratios = compute_ratios(spi_tbs, nt_coefficients)

    conc = compute_nt_conc(spi_tbs, nt_coefficients, ratios)

    # Set invalid tbs and weather-filtered values
    conc = apply_invalid_tbs_mask(conc, spi_tbs)
    conc = apply_weather_filter(conc, ratios, gr_thresholds)

    conc_int16 = conc.astype(np.int16)

    # This "conc_int16" field is identical to that saved to:
    #    ../nt_orig/system/new_2_iceconcentrations/
    #  eg
    #    nssss1d17tcon2018001
    # conc_int16.tofile('conc_raw_py.dat')

    # Apply NT-land spillover filter
    conc_spill = apply_nt_spillover(
        conc_int16=conc_int16, shoremap=shoremap, minic=minic
    )
    # Apply SST-threshold
    conc = apply_invalid_icemask(conc=conc_spill, invalid_ice_mask=invalid_ice_mask)

    ds = xr.Dataset({'conc': (('y', 'x'), conc)})

    return ds
