import datetime as dt

import numpy as np
import numpy.typing as npt
from pm_tb_data._types import Hemisphere
from pm_tb_data.fetch.amsr.au_si import get_au_si_tbs
from pm_tb_data.fetch.amsr.util import AMSR_RESOLUTIONS

from pm_icecon.interpolation import spatial_interp_tbs
from pm_icecon.nt.compute_nt_ic import goddard_nasateam
from pm_icecon.nt.params import get_cdr_nt_params
from pm_icecon.nt.tiepoints import get_tiepoints


def amsr2_goddard_nasateam(
    *,
    date: dt.date,
    hemisphere: Hemisphere,
    resolution: AMSR_RESOLUTIONS,
    invalid_ice_mask: npt.NDArray[np.bool_],
    # The shoremap has values 1-5 that indicate land,
    # coast, and cells away from coast (3-5).
    shoremap: npt.NDArray,
    # minic == minimum ice concentration grid.
    minic: npt.NDArray,
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

    nt_params = get_cdr_nt_params(
        hemisphere=hemisphere,
        platform="u2",
    )

    conc_ds = goddard_nasateam(
        tb_v19=spatial_interp_tbs(xr_tbs["v18"].data),
        tb_v37=spatial_interp_tbs(xr_tbs["v36"].data),
        tb_v22=spatial_interp_tbs(xr_tbs["v23"].data),
        tb_h19=spatial_interp_tbs(xr_tbs["h18"].data),
        shoremap=shoremap,
        minic=minic,
        invalid_ice_mask=invalid_ice_mask,
        gradient_thresholds=nt_params.gradient_thresholds,
        tiepoints=get_tiepoints(satellite="u2", hemisphere=hemisphere),
    )

    return conc_ds
