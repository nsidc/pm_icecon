import datetime as dt

from pm_tb_data.fetch.au_si import AU_SI_RESOLUTIONS, get_au_si_tbs

from pm_icecon._types import Hemisphere
from pm_icecon.bt.masks import get_ps_invalid_ice_mask
from pm_icecon.interpolation import spatial_interp_tbs
from pm_icecon.nt.compute_nt_ic import goddard_nasateam
from pm_icecon.nt.params.amsr2 import get_amsr2_params
from pm_icecon.nt.tiepoints import get_tiepoints


def amsr2_goddard_nasateam(
    *, date: dt.date, hemisphere: Hemisphere, resolution: AU_SI_RESOLUTIONS
):
    """Compute sea ice concentration from AU_SI25 TBs.

    Utilizes the bootstrap algorithm as organized by the original code from
    GSFC.
    """
    xr_tbs = get_au_si_tbs(
        date=date,
        hemisphere=hemisphere,
        resolution=resolution,
    )

    nt_params = get_amsr2_params(
        hemisphere=hemisphere,
        resolution=resolution,
    )

    # TODO: this function is currently defined in the bootstrap-specific masks
    # module. Should it be moved to the top-level masks? Originally split masks
    # between nt and bt modules because the original goddard nasateam example
    # used a unique invalid ice mask. Eventually won't matter too much because
    # we plan to move most masks into common nc files that will be read on a
    # per-grid basis.
    invalid_ice_mask = get_ps_invalid_ice_mask(
        hemisphere=hemisphere,
        date=date,
        resolution=resolution,
    )

    conc_ds = goddard_nasateam(
        tb_v19=spatial_interp_tbs(xr_tbs["v18"].data),
        tb_v37=spatial_interp_tbs(xr_tbs["v36"].data),
        tb_v22=spatial_interp_tbs(xr_tbs["v23"].data),
        tb_h19=spatial_interp_tbs(xr_tbs["h18"].data),
        shoremap=nt_params.shoremap,
        minic=nt_params.minic,
        invalid_ice_mask=invalid_ice_mask,
        gradient_thresholds=nt_params.gradient_thresholds,
        tiepoints=get_tiepoints(satellite="u2", hemisphere=hemisphere),
    )

    return conc_ds
