import numpy as np
from numpy.testing import assert_equal

# from cdr_amsr2.tests.regression.util import REGRESSION_DATA_DIR
from cdr_amsr2.constants import PACKAGE_DIR
from cdr_amsr2.nt.api import original_example


def test_nt_f17_regression():
    """Regressi5on test for NT F17 output."""
    regression_data = np.fromfile(
        # TODO: move regression data to regression data dir location
        # REGRESSION_DATA_DIR / 'nt_f17_regression/...',
        PACKAGE_DIR / 'nt' / 'nt_sample_nh.dat',
        dtype=np.int16,
    ).reshape((448, 304))

    actual_ds = original_example()

    assert_equal(
        regression_data,
        actual_ds.conc.data,
    )
