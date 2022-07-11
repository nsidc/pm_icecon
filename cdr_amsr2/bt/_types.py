from typing import Optional, TypedDict


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


class Variables(TypedDict):
    adoff: float
    itp: list[float]  # len 2
    itp2: list[float]  # len 2
    radlen1: float
    radlen2: float
    radoff1: float
    radoff2: float
    radslp1: float
    radslp2: float
    v1937: list[float]  # len 2
    vh37: list[float]  # len 2
    wtp: list[float]  # len 2
    wtp2: list[float]  # len 2

    wtp19v: Optional[float]
    wtp37v: Optional[float]
    wtp37h: Optional[float]
