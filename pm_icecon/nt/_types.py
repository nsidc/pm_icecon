from typing import TypedDict

import numpy.typing as npt


class NasateamCoefficients(TypedDict):
    A: float
    B: float
    C: float
    D: float
    E: float
    F: float
    G: float
    H: float
    I: float
    J: float
    K: float
    L: float


class NasateamGradientRatios(TypedDict):
    # 37v vs19v
    gr_3719: npt.NDArray
    # 22v vs 19v
    gr_2219: npt.NDArray
    # 19v vs 19h
    pr_1919: npt.NDArray
