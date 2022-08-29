import datetime as dt
from pathlib import Path

import numpy as np
import numpy.typing as npt
import xarray as xr

import cdr_amsr2.bt.compute_bt_ic as bt
from cdr_amsr2._types import Hemisphere
from cdr_amsr2.bt.masks import get_ps_valid_ice_mask
from cdr_amsr2.bt.params.a2l1c import A2L1C_NORTH_PARAMS
from cdr_amsr2.bt.params.amsr2 import AMSR2_NORTH_PARAMS, AMSR2_SOUTH_PARAMS
from cdr_amsr2.bt.params.dmsp import F17_F18_NORTH_PARAMS
from cdr_amsr2.config.models.bt import BootstrapParams
from cdr_amsr2.constants import PACKAGE_DIR
from cdr_amsr2.fetch.a2l1c_625 import get_a2l1c_625_tbs
from cdr_amsr2.fetch.au_si import AU_SI_RESOLUTIONS, get_au_si_tbs
from cdr_amsr2.masks import (
    get_e2n625_land_mask,
    get_ps_land_mask,
    get_ps_pole_hole_mask,
)


def amsr2_bootstrap(
    *, date: dt.date, hemisphere: Hemisphere, resolution: AU_SI_RESOLUTIONS
) -> xr.Dataset:
    """Compute sea ice concentration from AU_SI TBs."""
    xr_tbs = get_au_si_tbs(
        date=date,
        hemisphere=hemisphere,
        resolution=resolution,
    )

    params = BootstrapParams(
        sat='u2',
        land_mask=get_ps_land_mask(hemisphere=hemisphere, resolution=resolution),
        # There's no pole hole in the southern hemisphere.
        pole_mask=(
            get_ps_pole_hole_mask(resolution=resolution)
            if hemisphere == 'north'
            else None
        ),
        valid_ice_mask=get_ps_valid_ice_mask(
            hemisphere=hemisphere,
            date=date,
            resolution=resolution,  # type: ignore[arg-type]
        ),
        **(AMSR2_NORTH_PARAMS if hemisphere == 'north' else AMSR2_SOUTH_PARAMS),
    )

    tbs = {
        'v19': xr_tbs['v18'].data,
        'v37': xr_tbs['v36'].data,
        'h37': xr_tbs['h36'].data,
        'v22': xr_tbs['v23'].data,
    }

    conc_ds = bt.bootstrap(
        tbs=tbs,
        params=params,
        date=date,
        hemisphere=hemisphere,
        resolution=resolution,
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

    sst_fn = Path(
        f'/share/apps/amsr2-cdr/bootstrap_masks/valid_seaice_e2n6.25_{date:%m}.dat'
    )
    sst_mask = np.fromfile(sst_fn, dtype=np.uint8).reshape(1680, 1680)
    is_high_sst = sst_mask == 50

    params = BootstrapParams(
        sat='a2l1c',
        land_mask=get_e2n625_land_mask(),
        # TODO: For now, let's NOT impose a pole hole on the A2L1C data
        pole_mask=None,
        valid_ice_mask=is_high_sst,
        **A2L1C_NORTH_PARAMS,
    )

    tbs = {
        'v19': xr_tbs['v18'].data,
        'v37': xr_tbs['v36'].data,
        'h37': xr_tbs['h36'].data,
        'v22': xr_tbs['v23'].data,
    }

    conc_ds = bt.bootstrap(
        tbs=tbs,
        params=params,
        date=date,
        hemisphere=hemisphere,
        resolution='6.25',
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
    resolution: AU_SI_RESOLUTIONS = '25'
    date = dt.date(2018, 2, 17)
    hemisphere: Hemisphere = 'north'
    params = BootstrapParams(
        sat='18_class',
        land_mask=get_ps_land_mask(hemisphere=hemisphere, resolution=resolution),
        pole_mask=get_ps_pole_hole_mask(resolution=resolution),
        valid_ice_mask=get_ps_valid_ice_mask(
            hemisphere=hemisphere,
            date=date,
            resolution=resolution,  # type: ignore[arg-type]
        ),
        **F17_F18_NORTH_PARAMS,
    )

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

        return bt.fdiv(raw.astype(np.float32), 10)

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
        date=date,
        hemisphere=hemisphere,
        resolution=resolution,
    )

    return conc_ds
