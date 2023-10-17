import numpy as np
from numpy.testing import assert_equal

from pm_icecon.fill_polehole import fill_pole_hole


def test_fill_pole_hole():
    conc = np.array(
        [
            [0, 97.2, 92.3, 0],
            [0, np.nan, np.nan, 0],
            [0, np.nan, np.nan, 0],
            [0, 99.0, 98.3, 0],
        ]
    )

    pole_mask = np.array(
        [
            [False, True, True, False],
            [False, True, True, False],
            [False, True, True, False],
            [False, True, True, False],
        ]
    )

    avg_in_mask = np.mean([97.2, 92.3, 99.0, 98.3])
    expected = np.array(
        [
            [0, 97.2, 92.3, 0],
            [0, avg_in_mask, avg_in_mask, 0],
            [0, avg_in_mask, avg_in_mask, 0],
            [0, 99.0, 98.3, 0],
        ]
    )

    actual = fill_pole_hole(conc=conc, near_pole_hole_mask=pole_mask)

    assert_equal(expected, actual)
