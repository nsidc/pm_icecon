import datetime as dt
from pathlib import Path

import numpy as np
import xarray as xr
from pm_tb_data.fetch.a2l1c_625 import get_a2l1c_625_tbs
from pm_tb_data.fetch.au_si import AU_SI_RESOLUTIONS, get_au_si_tbs

import pm_icecon.bt.compute_bt_ic as bt
from pm_icecon._types import Hemisphere
from pm_icecon.bt.params.ausi_amsr2 import get_amsr2_params
from pm_icecon.bt.params.cetbv2_amsr2 import A2L1C_NORTH_PARAMS
from pm_icecon.config.models.bt import BootstrapParams
from pm_icecon.interpolation import spatial_interp_tbs
from pm_icecon.masks import get_e2n625_land_mask


def amsr2_goddard_bootstrap(
    *, date: dt.date, hemisphere: Hemisphere, resolution: AU_SI_RESOLUTIONS
) -> xr.Dataset:
    """Compute sea ice concentration from AU_SI TBs.

    Utilizes the bootstrap algorithm as organized by the original code from
    GSFC.
    """
    xr_tbs = get_au_si_tbs(
        date=date,
        hemisphere=hemisphere,
        resolution=resolution,
    )

    params = get_amsr2_params(
        date=date,
        hemisphere=hemisphere,
        resolution=resolution,
    )

    conc_ds = bt.goddard_bootstrap(
        tb_v37=spatial_interp_tbs(xr_tbs["v36"].data),
        tb_h37=spatial_interp_tbs(xr_tbs["h36"].data),
        tb_v19=spatial_interp_tbs(xr_tbs["v18"].data),
        tb_v22=spatial_interp_tbs(xr_tbs["v23"].data),
        params=params,
        date=date,
    )

    return conc_ds


def a2l1c_goddard_bootstrap(
    *,
    date: dt.date,
    hemisphere: Hemisphere,
    tb_dir: Path,
    anc_dir: Path,
    ncfn_template: str,
    timeframe: str,
) -> xr.Dataset:
    """Compute sea ice concentration from L1C 6.25km TBs.

    Utilizes the bootstrap algorithm as organized by the original code from
    GSFC.
    """
    if hemisphere == "south":
        raise NotImplementedError("Southern hemisphere is not currently supported.")

    xr_tbs = get_a2l1c_625_tbs(
        base_dir=tb_dir,
        date=date,
        hemisphere="north",
        ncfn_template=ncfn_template,
        timeframe=timeframe,
    )

    sst_fn = anc_dir / f"valid_seaice_e2n6.25_{date:%m}.dat"
    sst_mask = np.fromfile(sst_fn, dtype=np.uint8).reshape(1680, 1680)
    is_high_sst = sst_mask == 50

    params = BootstrapParams(
        land_mask=get_e2n625_land_mask(anc_dir),
        pole_mask=None,
        invalid_ice_mask=is_high_sst,
        **A2L1C_NORTH_PARAMS,  # type: ignore
    )

    conc_ds = bt.goddard_bootstrap(
        tb_v37=spatial_interp_tbs(xr_tbs["v36"].data),
        tb_h37=spatial_interp_tbs(xr_tbs["h36"].data),
        tb_v19=spatial_interp_tbs(xr_tbs["v18"].data),
        tb_v22=spatial_interp_tbs(xr_tbs["v23"].data),
        params=params,
        date=date,
    )

    return conc_ds
