import datetime as dt
from pathlib import Path

import numpy as np
import xarray as xr

import cdr_amsr2.bt.compute_bt_ic as bt
from cdr_amsr2._types import Hemisphere
from cdr_amsr2.bt.masks import get_ps_invalid_ice_mask
from cdr_amsr2.bt.params.a2l1c import A2L1C_NORTH_PARAMS
from cdr_amsr2.bt.params.amsr2 import AMSR2_NORTH_PARAMS, AMSR2_SOUTH_PARAMS
from cdr_amsr2.config.models.bt import BootstrapParams
from cdr_amsr2.constants import BOOTSTRAP_MASKS_DIR
from cdr_amsr2.fetch.a2l1c_625 import get_a2l1c_625_tbs
from cdr_amsr2.fetch.au_si import AU_SI_RESOLUTIONS, get_au_si_tbs
from cdr_amsr2.interpolation import spatial_interp_tbs
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
        invalid_ice_mask=get_ps_invalid_ice_mask(
            hemisphere=hemisphere,
            date=date,
            resolution=resolution,  # type: ignore[arg-type]
        ),
        **(AMSR2_NORTH_PARAMS if hemisphere == 'north' else AMSR2_SOUTH_PARAMS),
    )

    conc_ds = bt.bootstrap(
        tb_v37=spatial_interp_tbs(xr_tbs['v36'].data),
        tb_h37=spatial_interp_tbs(xr_tbs['h36'].data),
        tb_v19=spatial_interp_tbs(xr_tbs['v18'].data),
        tb_v22=spatial_interp_tbs(xr_tbs['v23'].data),
        params=params,
        date=date,
        hemisphere=hemisphere,
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

    sst_fn = BOOTSTRAP_MASKS_DIR / f'valid_seaice_e2n6.25_{date:%m}.dat'
    sst_mask = np.fromfile(sst_fn, dtype=np.uint8).reshape(1680, 1680)
    is_high_sst = sst_mask == 50

    params = BootstrapParams(
        sat='a2l1c',
        land_mask=get_e2n625_land_mask(),
        # TODO: For now, let's NOT impose a pole hole on the A2L1C data
        pole_mask=None,
        invalid_ice_mask=is_high_sst,
        **A2L1C_NORTH_PARAMS,
    )

    conc_ds = bt.bootstrap(
        tb_v37=spatial_interp_tbs(xr_tbs['v36'].data),
        tb_h37=spatial_interp_tbs(xr_tbs['h36'].data),
        tb_v19=spatial_interp_tbs(xr_tbs['v18'].data),
        tb_v22=spatial_interp_tbs(xr_tbs['v23'].data),
        params=params,
        date=date,
        hemisphere=hemisphere,
    )

    return conc_ds
