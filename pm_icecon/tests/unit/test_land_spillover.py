import numpy as np
from numpy.testing import assert_equal

from pm_icecon.land_spillover import apply_nt2_land_spillover, create_land90


def test_apply_nt2_land_spillover():
    """Test the NASA Team 2 land spillover routine.

    TODO: this test and the `expected` array is constructed from the `actual`
    output, without much thought. This test currently tests the implementation
    hasn't changed (for this test case) but _not_ the logic of the code
    itself. We should also seprately test the logic of `create_land90`.
    """
    adj123 = np.array(
        [
            [0, 1, 2, 3, 100, 100, 100],
            [0, 1, 2, 3, 100, 100, 100],
            [0, 1, 2, 3, 100, 100, 100],
            [0, 1, 2, 3, 100, 100, 100],
            [0, 1, 2, 3, 100, 100, 100],
            [0, 1, 2, 3, 100, 100, 100],
            [0, 1, 2, 3, 100, 100, 100],
        ],
    )

    lc90 = create_land90(adj123=adj123)

    _land = 254
    conc = np.array(
        [
            [_land, 3, 0, 15, 15, 45, 50],
            [_land, 0, 0, 15, 15, 35, 55],
            [_land, 95, 0, 25, 15, 30, 68],
            [_land, 33, 45, 15, 30, 20, 45],
            [_land, 0, 80, 20, 0, 0, 60],
            [_land, 15, 3, 50, 0, 0, 0],
            [_land, 100, 3, 0, 0, 0, 0],
        ],
    )

    actual = apply_nt2_land_spillover(
        conc=conc,
        adj123=adj123,
        l90c=lc90,
    )

    expected = np.array(
        [
            [_land, 0, 0, 15, 15, 45, 50],
            [_land, 0, 0, 15, 15, 35, 55],
            [_land, 95, 0, 25, 15, 30, 68],
            [_land, 33, 45, 15, 30, 20, 45],
            [_land, 0, 80, 20, 0, 0, 60],
            [_land, 0, 0, 50, 0, 0, 0],
            [_land, 100, 0, 0, 0, 0, 0],
        ]
    )

    assert_equal(actual, expected)
