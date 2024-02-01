"""Parameters for use with AMSR2 (AU_SI{12|25}) data."""
from dataclasses import dataclass

from loguru import logger
from pm_tb_data._types import Hemisphere

from pm_icecon.nt._types import NasateamGradientRatioThresholds
from pm_icecon.nt.params.gradient_thresholds import get_cdr_rss_thresholds
from pm_icecon.nt.tiepoints import NasateamTiePoints, get_tiepoints


@dataclass
class NasateamParams:
    tiepoints: NasateamTiePoints
    gradient_thresholds: NasateamGradientRatioThresholds


def get_amsr2_params(
    *,
    hemisphere: Hemisphere,
) -> NasateamParams:
    # Get tiepoints
    nt_tiepoints = get_tiepoints(satellite="u2", hemisphere=hemisphere)

    # Gradient thresholds
    nt_gradient_thresholds = get_cdr_rss_thresholds(
        hemisphere=hemisphere, platform="f17"
    )
    logger.info("NT gradient threshold values for AMSR2 are copied from f17")

    return NasateamParams(
        tiepoints=nt_tiepoints,
        gradient_thresholds=nt_gradient_thresholds,
    )
