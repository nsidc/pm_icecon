"""Parameters for use with AMSR2 (AU_SI{12|25}) data."""
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from loguru import logger

from pm_icecon._types import Hemisphere
from pm_icecon.constants import CDR_TESTDATA_DIR
from pm_icecon.fetch.au_si import AU_SI_RESOLUTIONS
from pm_icecon.nt._types import NasateamGradientRatioThresholds
from pm_icecon.nt.params.goddard_rss import (
    RSS_F17_NORTH_GRADIENT_THRESHOLDS,
    RSS_F17_SOUTH_GRADIENT_THRESHOLDS,
)
from pm_icecon.nt.tiepoints import NasateamTiePoints, get_tiepoints
from pm_icecon.util import get_ps_grid_shape


@dataclass
class NasateamParams:
    shoremap: npt.NDArray
    minic: npt.NDArray
    tiepoints: NasateamTiePoints
    gradient_thresholds: NasateamGradientRatioThresholds


def get_amsr2_params(
    *,
    hemisphere: Hemisphere,
    resolution: AU_SI_RESOLUTIONS,
) -> NasateamParams:
    # Nasateam specific config
    _nasateam_ancillary_dir = CDR_TESTDATA_DIR / 'nasateam_ancillary'
    # TODO: type for shoremap? The shoremap has values 1-5 that indicate land,
    # coast, and cells away from coast (3-5)
    nt_shoremap = np.fromfile(
        (_nasateam_ancillary_dir / f'shoremap_amsru_{hemisphere[0]}h{resolution}.dat'),
        dtype=np.uint8,
    ).reshape(get_ps_grid_shape(hemisphere=hemisphere, resolution=resolution))
    # minic == minimum ice concentration grid. Used in the nasateam land
    # spillover code.
    # TODO: better description/type for minic.
    nt_minic = np.fromfile(
        (_nasateam_ancillary_dir / f'minic_amsru_{hemisphere[0]}h{resolution}.dat'),
        dtype=np.int16,
    ).reshape(get_ps_grid_shape(hemisphere=hemisphere, resolution=resolution))
    # Scale down by 10. The original alg. dealt w/ concentrations scaled by 10.
    nt_minic = nt_minic / 10  # type: ignore[assignment]

    # Get tiepoints
    nt_tiepoints = get_tiepoints(satellite='u2', hemisphere=hemisphere)

    # Gradient thresholds
    nt_gradient_thresholds = (
        RSS_F17_NORTH_GRADIENT_THRESHOLDS
        if hemisphere == 'north'
        else RSS_F17_SOUTH_GRADIENT_THRESHOLDS
    )
    logger.warning('NT gradient threshold values for AMSR2 are copied from f17_final')

    return NasateamParams(
        shoremap=nt_shoremap,
        minic=nt_minic,
        tiepoints=nt_tiepoints,
        gradient_thresholds=nt_gradient_thresholds,
    )
