from dataclasses import dataclass

from loguru import logger
from pm_tb_data._types import Hemisphere

from pm_icecon._types import ValidSatellites
from pm_icecon.nt._types import NasateamGradientRatioThresholds
from pm_icecon.nt.params.gradient_thresholds import get_cdr_rss_thresholds
from pm_icecon.nt.tiepoints import NasateamTiePoints, get_tiepoints


@dataclass
class NasateamParams:
    tiepoints: NasateamTiePoints
    gradient_thresholds: NasateamGradientRatioThresholds


def get_cdr_nt_params(
    *,
    hemisphere: Hemisphere,
    platform: ValidSatellites | str,
) -> NasateamParams:
    nt_tiepoints = get_tiepoints(satellite=platform, hemisphere=hemisphere)

    # Gradient thresholds
    if platform == "am2":
        logger.info("NT gradient threshold values for AMSR2 are copied from f17")
        nt_gradient_thresholds = get_cdr_rss_thresholds(
            hemisphere=hemisphere,
            platform="f17",
        )
    else:
        nt_gradient_thresholds = get_cdr_rss_thresholds(
            hemisphere=hemisphere,
            platform=platform,
        )

    return NasateamParams(
        tiepoints=nt_tiepoints,
        gradient_thresholds=nt_gradient_thresholds,
    )
