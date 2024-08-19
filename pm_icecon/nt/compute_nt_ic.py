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
from pm_icecon.nt._types import (
    NasateamCoefficients,
    NasateamGradientRatioThresholds,
    NasateamRatio,
)
from pm_icecon.nt.tiepoints import NasateamTiePoints


def compute_nt_coefficients(tp: NasateamTiePoints) -> NasateamCoefficients:
    """Compute coefficients for the NT algorithm.

    tp are the tiepoints, a dictionary of structure:
      tp[channel][tiepoint]
         where channel is 'v19', 'h19', 'v37'
               tiepoint is 'ow', 'my', 'fy'
                        for open water, multiyear, first-year respectively
    """
    # Intermediate variables
    tp_names = Literal["ow", "fy", "my"]
    diff: dict[tp_names, dict[str, float]] = {}
    sums: dict[tp_names, dict[str, float]] = {}
    for tp_name in ("ow", "fy", "my"):
        # This cast is necessary because mypy just sees that tp_name is a value
        # that takes a str. Dumb...
        tp_name = cast(tp_names, tp_name)

        diff[tp_name] = {}
        diff[tp_name]["19v19h"] = tp["19v"][tp_name] - tp["19h"][tp_name]
        diff[tp_name]["37v19v"] = tp["37v"][tp_name] - tp["19v"][tp_name]

        sums[tp_name] = {}
        sums[tp_name]["19v19h"] = tp["19v"][tp_name] + tp["19h"][tp_name]
        sums[tp_name]["37v19v"] = tp["37v"][tp_name] + tp["19v"][tp_name]

    coef_a = (
        diff["my"]["19v19h"] * diff["ow"]["37v19v"]
        - diff["my"]["37v19v"] * diff["ow"]["19v19h"]
    )

    coef_b = (
        diff["my"]["37v19v"] * sums["ow"]["19v19h"]
        - diff["ow"]["37v19v"] * sums["my"]["19v19h"]
    )

    coef_c = (
        diff["ow"]["19v19h"] * sums["my"]["37v19v"]
        - diff["my"]["19v19h"] * sums["ow"]["37v19v"]
    )

    coef_d = (
        sums["my"]["19v19h"] * sums["ow"]["37v19v"]
        - sums["my"]["37v19v"] * sums["ow"]["19v19h"]
    )

    coef_e = (
        diff["fy"]["19v19h"] * (diff["my"]["37v19v"] - diff["ow"]["37v19v"])
        + diff["ow"]["19v19h"] * (diff["fy"]["37v19v"] - diff["my"]["37v19v"])
        + diff["my"]["19v19h"] * (diff["ow"]["37v19v"] - diff["fy"]["37v19v"])
    )

    coef_f = (
        diff["fy"]["37v19v"] * (sums["my"]["19v19h"] - sums["ow"]["19v19h"])
        + diff["ow"]["37v19v"] * (sums["fy"]["19v19h"] - sums["my"]["19v19h"])
        + diff["my"]["37v19v"] * (sums["ow"]["19v19h"] - sums["fy"]["19v19h"])
    )

    coef_g = (
        diff["fy"]["19v19h"] * (sums["ow"]["37v19v"] - sums["my"]["37v19v"])
        + diff["ow"]["19v19h"] * (sums["my"]["37v19v"] - sums["fy"]["37v19v"])
        + diff["my"]["19v19h"] * (sums["fy"]["37v19v"] - sums["ow"]["37v19v"])
    )

    coef_h = (
        sums["fy"]["37v19v"] * (sums["ow"]["19v19h"] - sums["my"]["19v19h"])
        + sums["ow"]["37v19v"] * (sums["my"]["19v19h"] - sums["fy"]["19v19h"])
        + sums["my"]["37v19v"] * (sums["fy"]["19v19h"] - sums["ow"]["19v19h"])
    )

    coef_i = (
        diff["fy"]["37v19v"] * diff["ow"]["19v19h"]
        - diff["fy"]["19v19h"] * diff["ow"]["37v19v"]
    )

    coef_j = (
        diff["ow"]["37v19v"] * sums["fy"]["19v19h"]
        - diff["fy"]["37v19v"] * sums["ow"]["19v19h"]
    )

    coef_k = (
        sums["ow"]["37v19v"] * diff["fy"]["19v19h"]
        - diff["ow"]["19v19h"] * sums["fy"]["37v19v"]
    )

    coef_l = (
        sums["fy"]["37v19v"] * sums["ow"]["19v19h"]
        - sums["fy"]["19v19h"] * sums["ow"]["37v19v"]
    )

    coefs = NasateamCoefficients(
        A=coef_a,
        B=coef_b,
        C=coef_c,
        D=coef_d,
        E=coef_e,
        F=coef_f,
        G=coef_g,
        H=coef_h,
        I=coef_i,
        J=coef_j,
        K=coef_k,
        L=coef_l,
    )

    return coefs


def compute_ratio(tb1: npt.NDArray, tb2: npt.NDArray) -> NasateamRatio:
    tb_diff = tb1 - tb2
    tb_sum = tb1 + tb2
    tb_sum[tb_sum == 0] = 1  # Avoid div by zero
    ratio = NasateamRatio(np.divide(tb_diff, tb_sum))

    return ratio


def get_weather_filter_mask(
    *,
    gr_2219: NasateamRatio,
    gr_3719: NasateamRatio,
    gr_2219_threshold: float,
    gr_3719_threshold: float,
) -> npt.NDArray[np.bool_]:
    """Return a boolean array representing areas exceeding the given thresholds."""
    # fmt: off
    weather_filter_mask = (
        (gr_2219 > gr_2219_threshold)
        | (gr_3719 > gr_3719_threshold)
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


def apply_nt_spillover(
    *, conc: npt.NDArray, shoremap: npt.NDArray, minic: npt.NDArray
) -> npt.NDArray[np.int16]:
    """Apply the NASA Team land spillover routine."""
    """
      conc is 0 to > 100.0
        (per this code, negative value means missing data)

      shoremap is 0: ocean, 1: land, 2: coast,
                   3: coast, 4: near_coast, 5: far_coast
        in original, shoremap is big-endian int16, but nothing in the
          computation requires it to be anything other than int-like

      minic is 0 to 100.0
        (in original binary file, minic is 0-1000 which is conc * 10)
        but as included here, it is float64 with values 0 to 96.3 (=~100.0)

      It appears that there is an assumption that land is always >= 15% conc
      TODO: Perhaps this should be looking for ocean cells that are >= 15%?
    """
    newice = conc.copy()

    is_at_coast = shoremap == 3
    is_near_coast = shoremap == 4
    is_far_coastal = shoremap == 5

    mod_minic = minic.copy()
    mod_minic[is_at_coast & (minic > 60)] = 60
    mod_minic[is_near_coast & (minic > 40)] = 40
    mod_minic[is_far_coastal & (minic > 20)] = 20

    # Count number of nearby low ice conc
    n_low = np.zeros_like(conc, dtype=np.uint8)

    # TODO: The original NASA code allows low-conc values over land to
    #       count toward the number of nearby low-conc values needed to
    #       cause a grid cell to be considered spillover.  This seems like
    #       an error.
    # Note: This scheme does not work if the land values have
    #       no concentration value
    # n_low_nonland = np.zeros_like(conc, dtype=np.uint8)

    conc_equiv = conc.copy()
    # If the mean concentration value over the land is low, then
    # then it's probably been set to zero and should not be used
    # to determine whether spillover should be applied
    mean_concval_over_land = np.nanmean(conc[(shoremap == 1) | (shoremap == 2)])

    if mean_concval_over_land < 5:
        print("NT spillover is not counting low-conc land values")
        conc_equiv[shoremap == 1] = 100
        conc_equiv[shoremap == 2] = 100

    # The cdralgos version of the NT land spillover algorithm excludes
    # grid cells near the edge of the grid by looping over range (3 to dim-3)
    grid_edge = np.zeros(conc_equiv.shape, dtype=np.uint8)
    grid_edge[:3, :] = 1
    grid_edge[-3:, :] = 1
    grid_edge[:, :3] = 1
    grid_edge[:, -3:] = 1
    is_not_grid_edge = grid_edge == 0

    for joff in range(-3, 3 + 1):
        for ioff in range(-3, 3 + 1):
            offmax = max(abs(ioff), abs(joff))

            rolled = np.roll(conc_equiv, (joff, ioff), axis=(0, 1))

            is_rolled_low = (rolled < 15) & (rolled >= 0) & is_not_grid_edge

            if offmax <= 3:
                n_low[is_rolled_low & is_at_coast] += 1

            if offmax <= 2:
                n_low[is_rolled_low & is_near_coast] += 1

            if offmax <= 1:
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

    # This rounds the floating point values for easier comparison to v4 vals
    # which are scaled short ints
    newice = np.round(newice, 2)

    return newice


def _clamp_conc_and_set_flags(
    *,
    shoremap: npt.NDArray,
    conc: npt.NDArray,
) -> npt.NDArray:
    """Clap concentrations to a max of 100 and apply flag values.

    Currently just sets a land value. TODO: add coast flag value.
    TODO: Actually, we may want to remove *all* non-conc values so that
          there are no sentinel values in these fields.

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


def calc_nasateam_conc(
    *,
    pr_1919: NasateamRatio,
    gr_3719: NasateamRatio,
    tiepoints: NasateamTiePoints,
) -> npt.NDArray:
    """Return a sea ice concentration estimate at every grid cell.

    Concentrations are given as percentage (0-100+%). Concentrations can be >
    100%. In the Goddard formulation of the nasateam algorithm, concentrations
    above 100 get set to 100 _after_ spillover correction.
    """
    # Get gradient ratios and compute their product
    pr_gr_product = pr_1919 * gr_3719

    # Use tiepoints to compute algorithm coefficients and ...
    coefs = compute_nt_coefficients(tiepoints)

    dd = (
        coefs["E"]
        + coefs["F"] * pr_1919
        + coefs["G"] * gr_3719
        + coefs["H"] * pr_gr_product
    )

    fy = (
        coefs["A"]
        + coefs["B"] * pr_1919
        + coefs["C"] * gr_3719
        + coefs["D"] * pr_gr_product
    )

    my = (
        coefs["I"]
        + coefs["J"] * pr_1919
        + coefs["K"] * gr_3719
        + coefs["L"] * pr_gr_product
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


def goddard_nasateam(
    *,
    tb_v19: npt.NDArray,
    tb_v37: npt.NDArray,
    tb_v22: npt.NDArray,
    tb_h19: npt.NDArray,
    shoremap: npt.NDArray,
    minic: npt.NDArray,
    invalid_ice_mask: npt.NDArray[np.bool_],
    gradient_thresholds: NasateamGradientRatioThresholds,
    tiepoints: NasateamTiePoints,
) -> xr.Dataset:
    """NASA Team algorithm as organized by the orignal code from GSFC."""
    pr_1919 = compute_ratio(tb_v19, tb_h19)
    gr_3719 = compute_ratio(tb_v37, tb_v19)
    conc = calc_nasateam_conc(
        pr_1919=pr_1919,
        gr_3719=gr_3719,
        tiepoints=tiepoints,
    )

    # Set invalid tbs and weather-filtered values
    invalid_tb_mask = get_invalid_tbs_mask(
        tb_v19=tb_v19,
        tb_h19=tb_h19,
        tb_v37=tb_v37,
    )

    gr_2219 = compute_ratio(tb_v22, tb_v19)
    weather_filter_mask = get_weather_filter_mask(
        gr_2219=gr_2219,
        gr_3719=gr_3719,
        gr_2219_threshold=gradient_thresholds["2219"],
        gr_3719_threshold=gradient_thresholds["3719"],
    )
    conc[invalid_tb_mask | weather_filter_mask] = 0

    # Apply NT-land spillover filter
    conc = apply_nt_spillover(conc=conc, shoremap=shoremap, minic=minic)
    # Apply SST-threshold
    conc[invalid_ice_mask] = 0

    conc = _clamp_conc_and_set_flags(shoremap=shoremap, conc=conc)

    ds = xr.Dataset({"conc": (("y", "x"), conc)})

    return ds
