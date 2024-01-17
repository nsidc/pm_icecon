"""Parameters for use with NSIDC-0001 data."""
from dataclasses import dataclass

from loguru import logger
from pm_tb_data._types import Hemisphere

from pm_icecon.nt._types import NasateamGradientRatioThresholds
from pm_icecon.nt.params.goddard_rss import (
    RSS_F17_NORTH_GRADIENT_THRESHOLDS,
    RSS_F17_SOUTH_GRADIENT_THRESHOLDS,
)
from pm_icecon.nt.tiepoints import NasateamTiePoints, get_tiepoints


@dataclass
class NasateamParams:
    tiepoints: NasateamTiePoints
    gradient_thresholds: NasateamGradientRatioThresholds


def get_0001_nt_params(
    *,
    hemisphere: Hemisphere,
    platform: str,
) -> NasateamParams:
    # Get tiepoints
    # nt_tiepoints = get_tiepoints(satellite="u2", hemisphere=hemisphere)
    nt_tiepoints = get_tiepoints(satellite=platform, hemisphere=hemisphere)

    # Gradient thresholds
    nt_gradient_thresholds = (
        RSS_F17_NORTH_GRADIENT_THRESHOLDS
        if hemisphere == "north"
        else RSS_F17_SOUTH_GRADIENT_THRESHOLDS
    )
    logger.info("NT gradient threshold values for {platform} are copied from f17_final")

    return NasateamParams(
        tiepoints=nt_tiepoints,
        gradient_thresholds=nt_gradient_thresholds,
    )
