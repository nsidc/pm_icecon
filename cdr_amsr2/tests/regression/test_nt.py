import numpy as np
from numpy.testing import assert_equal

from cdr_amsr2.nt.api import original_example
from cdr_amsr2.tests.regression.util import REGRESSION_DATA_DIR


def test_nt_f17_regression():
    """Regression test for NT F17 output."""
    regression_data = np.fromfile(
        REGRESSION_DATA_DIR / 'nt_f17_regression' / 'nt_sample_nh.dat',
        dtype=np.int16,
    ).reshape((448, 304))

    actual_ds = original_example()

    assert_equal(
        regression_data,
        actual_ds.conc.data,
    )
