import datetime as dt
from pathlib import Path

import numpy as np
import numpy.typing as npt
import xarray as xr

import cdr_amsr2.bt.compute_bt_ic as bt
from cdr_amsr2._types import Hemisphere
from cdr_amsr2.config import import_cfg_file
from cdr_amsr2.config.models.bt import BootstrapParams
from cdr_amsr2.constants import PACKAGE_DIR
from cdr_amsr2.fetch.a2l1c_625 import get_a2l1c_625_tbs
from cdr_amsr2.fetch.au_si25 import get_au_si25_tbs

# Here, we manually-set the PSN25 land and pole masks

# Ocean has a value of 0, land a value of 1, and coast a value of 2.
_land_coast_array_psn25 = np.fromfile(
    (
        PACKAGE_DIR
        / '../legacy/SB2_NRT_programs'
        / '../SB2_NRT_programs/ANCILLARY/north_land_25'
    ).resolve(),
    dtype=np.int16,
).reshape(448, 304)

# TODO: land mask currently includes land and coast. Does this make sense? Are
# we ever going to need to coast values? Maybe rename to `LAND_COAST_MASK`?
LAND_MASK_psn25 = _land_coast_array_psn25 != 0

# values of 1 indicate the pole hole.
POLE_MASK_psn25 = (
    np.fromfile(
        (
            PACKAGE_DIR
            / '../legacy/SB2_NRT_programs'
            / '../SB2_NRT_programs/ANCILLARY/np_holemask.ssmi_f17'
        ).resolve(),
        dtype=np.int16,
    ).reshape(448, 304)
    == 1
)


# Here, we manually-set the E2N6.25 land and pole masks

# The authoritative mask for the NH EASE2 Arctic subset
# is a 1680x1680 (centered) subset of the full 2880x2880 EASE2 NH grid
# These data were derived from the MCD12Q1 land type data set
# Encoding is
#    50: Ocean (= water orthogonally connected to the global ocean)
#   100: Lake (= water that is not connected to the global ocean)
#   125: Coast (= "Land" that is adjacent to "Ocean"
#   150: Land ("land type 1" that is not water and not ice)
#   200: Ice (= land classification type "permanent ice"
# For the purposes of the Bootstrap algorithm:
#    50 --> 0  (= "ocean" where sea ice might occur)
#   all others ->
_land_coast_array_e2n625 = np.fromfile(
    Path('/share/apps/amsr2-cdr/bootstrap_masks/locli_e2n6.25_1680x1680.dat'),
    dtype=np.uint8,
).reshape(1680, 1680)

# TODO: land mask currently includes land and coast. Does this make sense? Are
# we ever going to need to coast values? Maybe rename to `LAND_COAST_MASK`?
LAND_MASK_e2n625 = _land_coast_array_e2n625 != 50


def amsr2_bootstrap(*, date: dt.date, hemisphere: Hemisphere) -> xr.Dataset:
    """Compute sea ice concentration from AU_SI25 TBs."""
    if hemisphere == 'south':
        raise NotImplementedError('Southern hemisphere is not currently supported.')

    xr_tbs = get_au_si25_tbs(
        base_dir=Path('/ecs/DP1/AMSA/AU_SI25.001/'),
        date=date,
        hemisphere='north',
    )

    params = BootstrapParams(
        sat='u2',
        land_mask=LAND_MASK_psn25,
        pole_mask=POLE_MASK_psn25,
    )

    variables = import_cfg_file(PACKAGE_DIR / 'bt' / 'ret_ic_variables_amsru.json')

    tbs = {
        'v19': xr_tbs['v18'].data,
        'v37': xr_tbs['v36'].data,
        'h37': xr_tbs['h36'].data,
        'v22': xr_tbs['v23'].data,
    }

    conc_ds = bt.bootstrap(
        tbs=tbs,
        params=params,
        variables=variables,
        date=date,
    )

    return conc_ds


def a2l1c_bootstrap(*, date: dt.date, hemisphere: Hemisphere) -> xr.Dataset:
    """Compute sea ice concentration from L1C 6.25km TBs."""
    if hemisphere == 'south':
        raise NotImplementedError('Southern hemisphere is not currently supported.')

    xr_tbs = get_a2l1c_625_tbs(
        base_dir=Path('/data/amsr2_subsets/'),
        date=date,
        hemisphere='north',
    )

    params = BootstrapParams(
        sat='a2l1c',
        land_mask=LAND_MASK_e2n625,
        # TODO: For now, let's NOT impose a pole hole on the A2L1C data
        pole_mask=None,
    )

    variables = import_cfg_file(PACKAGE_DIR / 'bt' / 'ret_ic_variables_amsru.json')

    tbs = {
        'v19': xr_tbs['v18'].data,
        'v37': xr_tbs['v36'].data,
        'h37': xr_tbs['h36'].data,
        'v22': xr_tbs['v23'].data,
    }

    conc_ds = bt.bootstrap(
        tbs=tbs,
        params=params,
        variables=variables,
        date=date,
    )

    return conc_ds


def original_f18_example() -> xr.Dataset:
    """Return concentration field example for f18_20180217.

    This example data does not perfectly match the outputs given by Goddard's
    code, but it is very close. A total of 4 cells differ 1.

    ```
    >>> exact[not_eq]
    array([984, 991, 975, 830], dtype=int16)
    >>> not_eq = exact != not_exact
    >>> not_exact[not_eq]
    array([983, 992, 974, 829], dtype=int16)
    ```

    the exact grid produced by the fortran code is in
    `legacy/SB2_NRT_programs/NH_20180217_SB2_NRT_f18.ic`
    """
    params = BootstrapParams(
        sat='18',
        land_mask=LAND_MASK_psn25,
        pole_mask=POLE_MASK_psn25,
    )
    variables = import_cfg_file(PACKAGE_DIR / 'bt' / 'ret_ic_variables.json')

    otbs: dict[str, npt.NDArray[np.float32]] = {}

    # TODO: read this data from a fetch operation.
    raw_fns = {
        'v19': '../SB2_NRT_programs/orig_input_tbs/tb_f18_20180217_nrt_n19v.bin',
        'h37': '../SB2_NRT_programs/orig_input_tbs/tb_f18_20180217_nrt_n37h.bin',
        'v37': '../SB2_NRT_programs/orig_input_tbs/tb_f18_20180217_nrt_n37v.bin',
        'v22': '../SB2_NRT_programs/orig_input_tbs/tb_f18_20180217_nrt_n22v.bin',
    }

    def _read_tb_field(tbfn: Path) -> npt.NDArray[np.float32]:
        # Read int16 scaled by 10 and return float32 unscaled
        raw = np.fromfile(tbfn, dtype=np.int16).reshape(448, 304)

        return raw.astype(np.float32) / 10

    for tb in ('v19', 'h37', 'v37', 'v22'):
        otbs[tb] = _read_tb_field(
            (
                PACKAGE_DIR
                / '../legacy/SB2_NRT_programs'
                / raw_fns[tb]  # type: ignore [literal-required]
            ).resolve()
        )

    conc_ds = bt.bootstrap(
        tbs=otbs,
        params=params,
        variables=variables,
        date=dt.date(2018, 2, 17),
    )

    return conc_ds
