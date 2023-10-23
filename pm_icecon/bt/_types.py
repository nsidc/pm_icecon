# For Pydantic ~1.9
from typing import NewType, TypedDict

# A tie point is a "canonical" brightness temperature for a given channel and
# H2O-phase. For bootstrap, the 3 relevant channels -- 19H, 37H, and 37V -- each
# have both an "ice" tiepoint and a "water" tiepoint. These tiepoints are
# brightness temperature values, and have units of Kelvins.
Tiepoint = NewType("Tiepoint", float)
# The Bootstrap algorithm uses combinations of these tiepoints, specifically the
# 37H and 37V tiepoints in the "vh37" tuple and the 19V and 37V tiepoints in the
# "v1937" tuple to indicate locations on a scatterplot of those channel's TB
# values with those water tie points where the pixel is calculated to have zero
# percent sea ice.
TiepointSet = tuple[Tiepoint, Tiepoint]


class Line(TypedDict):
    """A line (e.g., the AD line in the bootstrap alg).

    Note:
    The common formulation for a line is: `y=mx+b` where `m` is the `slope` and
    `b` is the offset.
    """

    offset: float
    slope: float
