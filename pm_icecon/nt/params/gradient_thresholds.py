"""Gradient thresholds for the nasateam weather filter.
"""

from pm_tb_data._types import NORTH, SOUTH

from pm_icecon.errors import UnexpectedSatelliteError
from pm_icecon.nt._types import NasateamGradientRatioThresholds

# These thresholds defined specifically for the CDR. These differ from thoes
# defined by Goddard.
# RSS is Remote Sensing Systems. Data from RSS is used for the Final CDR
# (g02202)
# TODO: These are referenced by platform.lower(), but the keys should be
#       valid RSS satellites
CDR_RSS_THRESHOLDS_NORTH = dict(
    f18=NasateamGradientRatioThresholds(
        {
            "3719": 0.050,
            "2219": 0.045,
        }
    ),
    f17=NasateamGradientRatioThresholds(
        {
            "3719": 0.050,
            "2219": 0.045,
        }
    ),
    f13=NasateamGradientRatioThresholds(
        {
            "3719": 0.050,
            "2219": 0.045,
        }
    ),
    f11=NasateamGradientRatioThresholds(
        {
            "3719": 0.050,
            "2219": 0.045,
        }
    ),
    f08=NasateamGradientRatioThresholds(
        {
            "3719": 0.050,
            "2219": 0.045,
        }
    ),
    n07=NasateamGradientRatioThresholds(
        {
            "3719": 0.07,
            # This value ensures the threshold is never met for SMMR.
            # TODO: Better way to express this? `None`?
            "2219": 9999.9,
        }
    ),
)

CDR_RSS_THRESHOLDS_SOUTH = dict(
    f18=NasateamGradientRatioThresholds(
        {
            "3719": 0.057,
            "2219": 0.045,
        }
    ),
    f17=NasateamGradientRatioThresholds(
        {
            "3719": 0.057,
            "2219": 0.045,
        }
    ),
    f13=NasateamGradientRatioThresholds(
        {
            "3719": 0.050,
            "2219": 0.045,
        }
    ),
    f11=NasateamGradientRatioThresholds(
        {
            "3719": 0.050,
            "2219": 0.045,
        }
    ),
    f08=NasateamGradientRatioThresholds(
        {
            "3719": 0.050,
            "2219": 0.045,
        }
    ),
    n07=NasateamGradientRatioThresholds(
        {
            "3719": 0.076,
            # This value ensures the threshold is never met for SMMR.
            # TODO: Better way to express this? `None`?
            "2219": 9999.9,
        }
    ),
)


def get_cdr_rss_thresholds(
    *, hemisphere, platform: str
) -> NasateamGradientRatioThresholds:
    """Get Gradient thresholds for the CDR.

    Note that goddard specific thresholds are defined and used in the `test_nt`
    regression test file.
    """
    rss_thresholds = {
        NORTH: CDR_RSS_THRESHOLDS_NORTH,
        SOUTH: CDR_RSS_THRESHOLDS_SOUTH,
    }[hemisphere]
    if platform not in rss_thresholds.keys():
        raise UnexpectedSatelliteError(
            f"No {hemisphere[0].upper()}H thresholds defined for {platform=}."
        )
    platform = platform.lower()

    return rss_thresholds[platform]
