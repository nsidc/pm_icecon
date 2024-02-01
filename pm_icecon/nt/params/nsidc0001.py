"""Parameters for use with NSIDC-0001 data."""

from loguru import logger
from pm_tb_data._types import Hemisphere
from pm_icecon._types import ValidSatellites

from pm_icecon.nt.params.gradient_thresholds import get_cdr_rss_thresholds
from pm_icecon.nt.tiepoints import get_tiepoints
from pm_icecon.nt.params.amsr2 import NasateamParams


def get_0001_nt_params(
    *,
    hemisphere: Hemisphere,
    platform: ValidSatellites | str,
) -> NasateamParams:
    nt_tiepoints = get_tiepoints(satellite=platform, hemisphere=hemisphere)

    # Gradient thresholds
    nt_gradient_thresholds = get_cdr_rss_thresholds(
        hemisphere=hemisphere, platform=platform
    )

    logger.info("NT gradient threshold values for {platform} are copied from f17_final")

    return NasateamParams(
        tiepoints=nt_tiepoints,
        gradient_thresholds=nt_gradient_thresholds,
    )
