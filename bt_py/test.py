import numpy as np
from numpy.testing import assert_equal

from bt_py.compute_bt_ic import tb_data_mask


def test_tb_data_mask():
    expected = np.array([1, 1, 0, 0], dtype=bool)

    actual = tb_data_mask(
        v37=np.array([1, 1, 0, 0], dtype=np.float32),
        h37=np.array([1, 0, 0, 0], dtype=np.float32),
        v19=np.array([1, 0, 0, 0], dtype=np.float32),
        v22=np.array([1, 1, 0, 0], dtype=np.float32),
        mintb=0.0,
        maxtb=0.9,
    )

    assert_equal(expected, actual)
