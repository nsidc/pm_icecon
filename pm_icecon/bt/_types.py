from typing import TypedDict

# A tiepoint is a point (x, y) on a scatterplot of Tbs between two channels.
# TODO: consider making this a `Point` type or dataclass that will make it clear
# that the first element is x and the second y.
# TODO: another thought/idea: maybe we can have the x and y dimensions take a
# name? E.g., x_name='37v' and y_name='37h' for wtp_37v37h? Or some other
# wrapper container that captures the association between two channels,
# tiepoints, and lnline>
Tiepoint = tuple[float, float]


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
