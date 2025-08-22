"""Code for making Tbs from different platforms consistent

This code aims to make Tbs from platforms other than F13 match F13.
"""

import copy
from typing import Literal

import numpy as np
import numpy.typing as npt

from pm_icecon.errors import UnexpectedSatelliteError

# RSS-specific Tb transformation values
# Each key is a platform name. This mapps to another dictionary of Tb channel
# names w/ maps to a tuple representing a slope and offset.
rss_tb_xfers: dict[str, dict[str, tuple[float, float]]] = dict(
    n07=dict(
        v37=(0.97901150, 6.6694386),
        h37=(1.0065762, 4.6935520),
        v19=(0.93256515, 25.874374),
    ),
    f08=dict(
        v37=(0.97513057, 5.5927164),
        h37=(0.97245795, 5.6908542),
        v19=(0.97687641, 5.8538911),
        v22=(0.96085082, 8.9943384),
    ),
    f11=dict(
        v37=(0.99609056, 0.69241640),
        h37=(0.99056213, 1.8797186),
        v19=(1.0039689, -0.88561701),
        v22=(1.0005052, -0.44676934),
    ),
    # Do nothing to the Tbs. These transformations are designed to make the
    # Tbs from other platforms match F13 as a baseline.
    f13=dict(),
    f17=dict(
        v37=(1.0224454, -6.5927872),
        h37=(0.99409019, 0.64754555),
        v19=(1.0388919, -6.5720982),
        v22=(1.0301712, -5.8031430),
    ),
    f18=dict(
        v37=(1.0164655, -5.2092480),
        h37=(0.98158527, 3.6840773),
        v19=(1.0327091, -6.2381073),
        v22=(1.0240212, -5.8834975),
    ),
)


def xfer_rss_tbs(
    *,
    tbs: dict,
    platform: str,
) -> dict[str, npt.NDArray[np.float32]]:
    """Transform RSS (final, NSIDC-0001 and NSIDC-0007) Tbs to match F13."""
    if platform not in rss_tb_xfers:
        raise UnexpectedSatelliteError(
            f"No transformation parameters exist for {platform=}"
        )

    xfrs = rss_tb_xfers[platform]

    transformed = copy.deepcopy(tbs)

    for channel, (slope, offset) in xfrs.items():
        if channel in transformed.keys():
            is_tbs_zero = transformed[channel] == 0
            transformed[channel] = (transformed[channel] * slope) + offset
            transformed[channel][is_tbs_zero] = 0

    return transformed


# TODO: make this function work similarily to `xfer_rss_tbs`. Implement other
# platforms. Currently only used for testing that the original F18 example
# provided by Goddard works.
def xfer_class_tbs(
    *,
    tb_v37: npt.NDArray,
    tb_h37: npt.NDArray,
    tb_v19: npt.NDArray,
    tb_v22: npt.NDArray,
    sat: Literal["f17", "f18"],
) -> dict[str, npt.NDArray[np.floating]]:
    """Transform selected CLASS (NRT) TBs for consistentcy with timeseries.

    Some CLASS data should be transformed via linear regression for consistenicy
    with F13.
    """
    is_v37_zeros = tb_v37 == 0
    is_h37_zeros = tb_h37 == 0
    is_v19_zeros = tb_v19 == 0
    is_v22_zeros = tb_v22 == 0

    # NRT regressions
    if sat == "f17":
        tb_v37 = (1.0170066 * tb_v37) + -4.9383355
        tb_h37 = (1.0009720 * tb_h37) + -1.3709822
        tb_v19 = (1.0140723 * tb_v19) + -3.4705583
        tb_v22 = (0.99652931 * tb_v22) + -0.82305684
    elif sat == "f18":
        tb_v37 = (1.0104497 * tb_v37) + -3.3174017
        tb_h37 = (0.98914390 * tb_h37) + 1.2031835
        tb_v19 = (1.0057373 * tb_v19) + -0.92638520
        tb_v22 = (0.98793409 * tb_v22) + 1.2108198
    else:
        raise UnexpectedSatelliteError(f"No such tb xform: {sat}")

    tb_v37[is_v37_zeros] = 0
    tb_h37[is_h37_zeros] = 0
    tb_v19[is_v19_zeros] = 0
    tb_v22[is_v22_zeros] = 0

    return {
        "tb_v37": tb_v37,
        "tb_h37": tb_h37,
        "tb_v19": tb_v19,
        "tb_v22": tb_v22,
    }
