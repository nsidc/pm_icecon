import numpy as np
from numpy.testing import assert_almost_equal

from cdr_amsr2.interpolation import spatial_interp_tbs


def test_spatial_interp_tbs():
    example_data = np.array(
        [
            [3, 2, 1],
            [1, 0, 3],
            [9, 2, 3],
        ],
        dtype=np.float32,
    )

    adjacent = np.array([2, 1, 3, 2])
    diagonal = np.array([3, 1, 9, 3])
    diagonal_weighted = diagonal * 0.707

    count = len(adjacent) + (len(diagonal) * 0.707)

    expected_value = np.concatenate([adjacent, diagonal_weighted]).sum() / count

    expected = np.array(
        [
            [3, 2, 1],
            [1, expected_value, 3],
            [9, 2, 3],
        ],
        dtype=np.float32,
    )

    actual = spatial_interp_tbs({'chan': example_data})['chan']

    assert_almost_equal(expected, actual, decimal=6)


def test_spatial_interp_tbs_edge():
    example_data = np.array(
        [
            [3, 2, 1],
            [1, 9, 0],
            [3, 2, 3],
        ],
        dtype=np.float32,
    )

    adjacent = np.array(
        [
            1,
            9,
            3,
        ]
    )
    diagonal = np.array(
        [
            2,
            2,
            # These next two values are actually adjacent to the Tbs but due to the
            # 'nearest' method used in the algorithm, get used as 'adjacent'.
            1,
            3,
        ]
    )
    diagonal_weighted = diagonal * 0.707

    count = len(adjacent) + (len(diagonal) * 0.707)

    expected_value = np.concatenate([adjacent, diagonal_weighted]).sum() / count

    expected = np.array(
        [
            [3, 2, 1],
            [1, 9, expected_value],
            [3, 2, 3],
        ],
        dtype=np.float32,
    )

    actual = spatial_interp_tbs({'chan': example_data})['chan']

    assert_almost_equal(expected, actual, decimal=6)


def test_spatial_interp_tbs_only_if_enough():
    """Test that interp does not occur when not enough data.

    The upper-left corner, which only has one diagonal connection, should not be
    interpolated.
    """
    example_data = np.array(
        [
            [0, 0, 1],
            [0, 1, 0],
            [1, 1, 1],
        ],
        dtype=np.float32,
    )

    expected = np.array(
        [
            [0, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
        ],
        dtype=np.float32,
    )

    actual = spatial_interp_tbs({'chan': example_data})['chan']

    assert_almost_equal(expected, actual, decimal=6)
