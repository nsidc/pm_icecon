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


NasateamGradientRatioThresholds = TypedDict(
    'NasateamGradientRatioThresholds',
    {
        '2219': float,
        '3719': float,
    },
)
