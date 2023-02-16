from typing import NewType, TypedDict

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


NasateamRatio = NewType('NasateamRatio', npt.NDArray)


class NasateamRatios(TypedDict):
    # gradient ratio for 22v vs 19v
    gr_2219: NasateamRatio

    # gradient ratio for 37v vs19v
    gr_3719: NasateamRatio

    # polarization ratio for 19v vs 19h
    pr_1919: NasateamRatio


NasateamGradientRatioThresholds = TypedDict(
    'NasateamGradientRatioThresholds',
    {
        '2219': float,
        '3719': float,
    },
)
