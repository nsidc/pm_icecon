import datetime as dt
from pathlib import Path

import numpy as np
import numpy.typing as npt
import xarray as xr
from pm_tb_data.fetch.a2l1c_625 import get_a2l1c_625_tbs
from pm_tb_data.fetch.au_si import AU_SI_RESOLUTIONS, get_au_si_tbs
from pm_tb_data._types import Hemisphere

import pm_icecon.bt.compute_bt_ic as bt
from pm_icecon.bt.params.ausi_amsr2 import get_amsr2_params
from pm_icecon.bt.params.cetbv2_amsr2 import A2L1C_NORTH_PARAMS
from pm_icecon.config.models.bt import BootstrapParams
from pm_icecon.interpolation import spatial_interp_tbs


def amsr2_goddard_bootstrap(
    *,
    date: dt.date,
    hemisphere: Hemisphere,
    resolution: AU_SI_RESOLUTIONS,
    land_mask: npt.NDArray[np.bool_],
    invalid_ice_mask: npt.NDArray[np.bool_],
    pole_mask: npt.NDArray[np.bool_] | None,
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
        land_mask=land_mask,
        invalid_ice_mask=invalid_ice_mask,
    )

    return conc_ds


def a2l1c_goddard_bootstrap(
    *,
    date: dt.date,
    hemisphere: Hemisphere,
    tb_dir: Path,
    land_mask: npt.NDArray[np.bool_],
    invalid_ice_mask: npt.NDArray[np.bool_],
    ncfn_template: str,
    timeframe: str,
) -> xr.Dataset:
    """Compute sea ice concentration from L1C 6.25km TBs.

    Utilizes the bootstrap algorithm as organized by the original code from
    GSFC.

    Note: an invalid seaice mask can be created from
    `valid_seaice_e2n6.25_{date:%m}.dat`. The value of 50 in this file
    represents an invalid mask:

    ```
    sst_fn = f"valid_seaice_e2n6.25_{date:%m}.dat"
    sst_mask = np.fromfile(sst_fn, dtype=np.uint8).reshape(1680, 1680)
    is_high_sst = sst_mask == 50
    ```
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

    params = BootstrapParams(
        **A2L1C_NORTH_PARAMS,  # type: ignore[arg-type]
    )

    conc_ds = bt.goddard_bootstrap(
        tb_v37=spatial_interp_tbs(xr_tbs["v36"].data),
        tb_h37=spatial_interp_tbs(xr_tbs["h36"].data),
        tb_v19=spatial_interp_tbs(xr_tbs["v18"].data),
        tb_v22=spatial_interp_tbs(xr_tbs["v23"].data),
        params=params,
        date=date,
        land_mask=land_mask,
        invalid_ice_mask=invalid_ice_mask,
    )

    return conc_ds
