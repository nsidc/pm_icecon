import datetime as dt

import numpy as np
from loguru import logger

from pm_icecon._types import Hemisphere
from pm_icecon.bt.masks import get_ps_invalid_ice_mask
from pm_icecon.constants import CDR_TESTDATA_DIR
from pm_icecon.fetch.au_si import AU_SI_RESOLUTIONS, get_au_si_tbs
from pm_icecon.interpolation import spatial_interp_tbs
from pm_icecon.nt.compute_nt_ic import goddard_nasateam
from pm_icecon.nt.params.goddard_rss import (
    RSS_F17_NORTH_GRADIENT_THRESHOLDS,
    RSS_F17_SOUTH_GRADIENT_THRESHOLDS,
)
from pm_icecon.nt.tiepoints import get_tiepoints
from pm_icecon.util import get_ps_grid_shape


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

    _nasateam_ancillary_dir = CDR_TESTDATA_DIR / 'nasateam_ancillary'
    shoremap = np.fromfile(
        (_nasateam_ancillary_dir / f'shoremap_amsru_{hemisphere[0]}h{resolution}.dat'),
        dtype=np.uint8,
    ).reshape(get_ps_grid_shape(hemisphere=hemisphere, resolution=resolution))
    minic = np.fromfile(
        (_nasateam_ancillary_dir / f'minic_amsru_{hemisphere[0]}h{resolution}.dat'),
        dtype=np.int16,
    ).reshape(get_ps_grid_shape(hemisphere=hemisphere, resolution=resolution))

    # Scale down by 10. The original alg. dealt w/ concentrations scaled by 10.
    minic = minic / 10  # type: ignore[assignment]

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

    # Use the gradient thresholds from RSS F17 for now.
    gradient_thresholds = (
        RSS_F17_NORTH_GRADIENT_THRESHOLDS
        if hemisphere == 'north'
        else RSS_F17_SOUTH_GRADIENT_THRESHOLDS
    )
    logger.warning(
        'The graident threshold values were stolen from f17_final!'
        ' Do we need new ones for AMSR2? How do we get them?'
    )

    conc_ds = goddard_nasateam(
        tb_v19=spatial_interp_tbs(xr_tbs['v18'].data),
        tb_v37=spatial_interp_tbs(xr_tbs['v36'].data),
        tb_v22=spatial_interp_tbs(xr_tbs['v23'].data),
        tb_h19=spatial_interp_tbs(xr_tbs['h18'].data),
        shoremap=shoremap,
        minic=minic,
        invalid_ice_mask=invalid_ice_mask,
        gradient_thresholds=gradient_thresholds,
        tiepoints=get_tiepoints(satellite='u2', hemisphere=hemisphere),
    )

    return conc_ds
