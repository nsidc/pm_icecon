from typing import NewType, TypedDict

# A tie point is a "canonical" brightness temperature for a given channel and
# H2O-phase. For bootstrap, the 3 relevant channels -- 19H, 37H, and 37V -- each
# have both an "ice" tiepoint and a "water" tiepoint. These tiepoints are
# brightness temperature values, and have units of Kelvins.
Tiepoint = NewType('Tiepoint', float)
# The Bootstrap algorithm uses combinations of these tiepoints, specifically the
# 37H and 37V tiepoints in the "vh37" tuple and the 19V and 37V tiepoints in the
# "v1937" tuple to indicate locations on a scatterplot of those channel's TB
# values with those water tie points where the pixel is calculated to have zero
# percent sea ice.
TiepointSet = tuple[Tiepoint, Tiepoint]


class RawFns(TypedDict):
    # maps strings to strings representing filepaths.
    h37: str
    land: str
    nphole: str
    v19: str
    v22: str
    v37: str


# Dict returned by `ret_para_nsb2`
class ParaVals(TypedDict):
    iceline: list[float]  # len 2
    itp: list[float]  # len 2
    lnchk: float
    lnline: list[float]  # len 2
    wintrc: float
    wslope: float
    wtp: list[float]  # len 2
    wxlimt: float


class Line(TypedDict):
    offset: float
    slope: float
