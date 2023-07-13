# For Pydantic ~1.9
from typing import NewType, TypedDict

import numpy.typing as npt

# For Pydantic ~2.0
# from typing import NewType
# from typing_extensions import TypedDict



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
