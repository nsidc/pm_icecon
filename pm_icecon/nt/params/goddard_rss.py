"""Parameter from Goddard for 'Final' / RSS data.

RSS is Remote Sensing Systems. Data from RSS is used for the Final CDR  (g02202)

The parameters defined in this file are not suitable for use with data from the
NOAA Comprehensive Large Array-Data Stewardship System (CLASS). Data from CLASS
is used for the 'NRT' CDR (g10016).
"""
from pm_icecon.nt._types import NasateamGradientRatioThresholds

RSS_F17_NORTH_GRADIENT_THRESHOLDS = NasateamGradientRatioThresholds(
    {
        "3719": 0.050,
        "2219": 0.045,
    }
)


RSS_F17_SOUTH_GRADIENT_THRESHOLDS = NasateamGradientRatioThresholds(
    {
        "3719": 0.053,
        "2219": 0.045,
    }
)
