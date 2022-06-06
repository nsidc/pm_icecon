import numpy as np
from numpy.testing import assert_equal

from cdr_amsr2.bt_py.compute_bt_ic import tb_data_mask


def test_tb_data_mask():
    expected = np.array([1, 1, 0, 0], dtype=bool)

    actual = tb_data_mask(
        tbs=(
            np.array([1, 1, 0, 0], dtype=np.float32),
            np.array([1, 0, 0, 0], dtype=np.float32),
            np.array([1, 0, 0, 0], dtype=np.float32),
            np.array([1, 1, 0, 0], dtype=np.float32),
        ),
        min_tb=0.0,
        max_tb=0.9,
    )

    assert_equal(expected, actual)
