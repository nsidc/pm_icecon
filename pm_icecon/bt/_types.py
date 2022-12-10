from typing import TypedDict

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