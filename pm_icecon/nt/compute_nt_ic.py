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

from typing import Literal, cast

import numpy as np
import numpy.typing as npt
import xarray as xr

from pm_icecon.constants import DEFAULT_FLAG_VALUES
from pm_icecon.nt.tiepoints import NasateamTiePoints


def fdiv(a, b):
    return np.divide(a, b, dtype=np.float32)


def compute_nt_coefficients(tp: NasateamTiePoints) -> dict[str, float]:
    """Compute coefficients for the NT algorithm.

    tp are the tiepoints, a dictionary of structure:
      tp[channel][tiepoint]
         where channel is 'v19', 'h19', 'v37'
               tiepoint is 'ow', 'my', 'fy'
                        for open water, multiyear, first-year respectively
    """
    # Intermediate variables
    tp_names = Literal['ow', 'fy', 'my']
    diff: dict[tp_names, dict[str, float]] = {}
    sums: dict[tp_names, dict[str, float]] = {}
    for tp_name in ('ow', 'fy', 'my'):
        # This cast is necessary because mypy just sees that tp_name is a value
        # that takes a str. Dumb...
        tp_name = cast(tp_names, tp_name)

        diff[tp_name] = {}
        diff[tp_name]['19v19h'] = tp['19v'][tp_name] - tp['19h'][tp_name]
        diff[tp_name]['37v19v'] = tp['37v'][tp_name] - tp['19v'][tp_name]

        sums[tp_name] = {}
        sums[tp_name]['19v19h'] = tp['19v'][tp_name] + tp['19h'][tp_name]
        sums[tp_name]['37v19v'] = tp['37v'][tp_name] + tp['19v'][tp_name]

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


def compute_ratios(
    *,
    tb_h19: npt.NDArray,
    tb_v19: npt.NDArray,
    tb_v22: npt.NDArray,
    tb_v37: npt.NDArray,
    coefs,
) -> dict[str, npt.NDArray]:
    """Return calculated gradient ratios.

    TODO: make this function more generic. There should be a func for computing
    a single gradient ratio. This function could call that.
    """
    ratios = {}

    dif_37v19v = tb_v37 - tb_v19
    sum_37v19v = tb_v37 + tb_v19
    sum_37v19v[sum_37v19v == 0] = 1  # Avoid div by zero
    ratios['gr_3719'] = np.divide(dif_37v19v, sum_37v19v)

    dif_22v19v = tb_v22 - tb_v19
    sum_22v19v = tb_v22 + tb_v19
    sum_22v19v[sum_22v19v == 0] = 1  # Avoid div by zero
    ratios['gr_2219'] = np.divide(dif_22v19v, sum_22v19v)

    dif_19v19h = tb_v19 - tb_h19
    sum_19v19h = tb_v19 + tb_h19
    sum_19v19h[sum_19v19h == 0] = 1  # Avoid div by zero
    ratios['pr_1919'] = np.divide(dif_19v19h, sum_19v19h)

    return ratios


def get_weather_filter_mask(
    *, ratios: dict[str, npt.NDArray], gr_thresholds: dict[str, float]
) -> npt.NDArray[np.bool_]:
    # Determine where array is weather-filtered
    print(f'gr_thresholds 2219: {gr_thresholds["2219"]}')
    print(f'gr_thresholds 3719: {gr_thresholds["3719"]}')

    # fmt: off
    weather_filter_mask = (
        (ratios['gr_2219'] > gr_thresholds['2219'])
        | (ratios['gr_3719'] > gr_thresholds['3719'])
    )
    # fmt: on

    return weather_filter_mask


# TODO: this function very similar to `tb_data_mask` in `compute_bt_ic`.
def get_invalid_tbs_mask(
    *,
    tb_v19: npt.NDArray,
    tb_h19: npt.NDArray,
    tb_v37: npt.NDArray,
) -> npt.NDArray[np.bool_]:
    is_valid_tbs = (tb_v19 > 0) & (tb_h19 > 0) & (tb_v37 > 0)
    invalid_tbs = ~is_valid_tbs

    return invalid_tbs


def compute_nt_conc(
    *,
    coefs: dict[str, float],
    ratios: dict[str, npt.NDArray],
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
    dd[dd == 0] = 0.01  # This causes the denominator to become 1.0

    conc = (fy + my) / dd * 100.0

    # Clamp concentrations to be above 0. Later (after applying the spillover
    # algorithm), concentrations will be clamped to 100 at the upper end. At
    # this point, concentrations may be > 100.
    conc[conc < 0] = 0

    return conc


def apply_nt_spillover(
    *, conc: npt.NDArray, shoremap: npt.NDArray, minic: npt.NDArray
) -> npt.NDArray[np.int16]:
    """Apply the NASA Team land spillover routine."""
    newice = conc.copy()

    is_at_coast = shoremap == 5
    is_near_coast = shoremap == 4
    is_far_coastal = shoremap == 3

    mod_minic = minic.copy()
    mod_minic[is_at_coast & (minic > 20)] = 20
    mod_minic[is_near_coast & (minic > 40)] = 40
    mod_minic[is_far_coastal & (minic > 60)] = 60

    # Count number of nearby low ice conc
    n_low = np.zeros_like(conc, dtype=np.uint8)

    for joff in range(-3, 3 + 1):
        for ioff in range(-3, 3 + 1):
            offmax = max(abs(ioff), abs(joff))

            rolled = np.roll(conc, (joff, ioff), axis=(0, 1))
            is_rolled_low = (rolled < 15) & (rolled >= 0)

            if offmax <= 1:
                n_low[is_rolled_low & is_at_coast] += 1

            if offmax <= 2:
                n_low[is_rolled_low & is_near_coast] += 1

            if offmax <= 3:
                n_low[is_rolled_low & is_far_coastal] += 1

    # Note: there are meaningless differences "at the edge" in these counts
    # because the spatial interpolation is not identical along the border

    where_reduce_ice = (n_low >= 3) & (shoremap > 2)
    newice[where_reduce_ice] -= mod_minic[where_reduce_ice]

    where_ice_overreduced = (conc >= 0) & (newice < 0) & (shoremap > 2)
    newice[where_ice_overreduced] = 0

    # Preserve missing data (conc value of -10)
    where_missing = (conc < 0) & where_reduce_ice & (shoremap > 2)
    newice[where_missing] = conc[where_missing]

    return newice


def _clamp_conc_and_set_flags(*, shoremap: npt.NDArray, conc: npt.NDArray):
    """Clap concentrations to a max of 100 and apply flag values.

    Currently just sets a land value. TODO: add coast flag value.

    We clamp concentrations to a max of 100 at this step instead of in
    `compute_nt_conc` because the original algorithm implemented clamping only
    after the land spillover correction is applied.
    """
    flagged_conc = conc.copy()
    # Clamp concentrations above 100 to 100.
    flagged_conc[flagged_conc > 100] = 100
    # Set flag values
    # Set land/coast flag values.
    # TODO: do we want the coast to be 'land' as it is in bootstrap?
    # 1 == land
    flagged_conc[shoremap == 1] = DEFAULT_FLAG_VALUES.land
    # 2 == coast
    # TODO: re-add this flag. For now, making the flags for nt consistent w/ bt.
    # newice[shoremap == 2] = DEFAULT_FLAG_VALUES.coast
    flagged_conc[shoremap == 2] = DEFAULT_FLAG_VALUES.land

    return flagged_conc


def nasateam(
    *,
    tb_v19: npt.NDArray,
    tb_v37: npt.NDArray,
    tb_v22: npt.NDArray,
    tb_h19: npt.NDArray,
    shoremap: npt.NDArray,
    minic: npt.NDArray,
    invalid_ice_mask: npt.NDArray[np.bool_],
    gradient_thresholds: dict[str, float],
    tiepoints: NasateamTiePoints,
):
    print(f'tiepoints: {tiepoints}')

    nt_coefficients = compute_nt_coefficients(tiepoints)
    print(f'NT coefs: {nt_coefficients}')
    for c in ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L'):
        print(f'  coef {c}: {nt_coefficients[c]}')

    ratios = compute_ratios(
        tb_h19=tb_h19,
        tb_v19=tb_v19,
        tb_v22=tb_v22,
        tb_v37=tb_v37,
        coefs=nt_coefficients,
    )

    conc = compute_nt_conc(
        coefs=nt_coefficients,
        ratios=ratios,
    )

    # Set invalid tbs and weather-filtered values
    invalid_tb_mask = get_invalid_tbs_mask(
        tb_v19=tb_v19,
        tb_h19=tb_h19,
        tb_v37=tb_v37,
    )
    weather_filter_mask = get_weather_filter_mask(
        ratios=ratios,
        gr_thresholds=gradient_thresholds,
    )
    conc[invalid_tb_mask | weather_filter_mask] = 0

    # Apply NT-land spillover filter
    conc = apply_nt_spillover(conc=conc, shoremap=shoremap, minic=minic)
    # Apply SST-threshold
    conc[invalid_ice_mask] = 0

    conc = _clamp_conc_and_set_flags(shoremap=shoremap, conc=conc)

    ds = xr.Dataset({'conc': (('y', 'x'), conc)})

    return ds
