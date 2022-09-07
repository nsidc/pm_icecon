import numpy as np
from numpy.testing import assert_equal

from cdr_amsr2.nt.api import original_example
from cdr_amsr2.tests.regression.util import REGRESSION_DATA_DIR
from cdr_amsr2.util import get_ps25_grid_shape


def _hack_flag_vals(conc):
    from cdr_amsr2.constants import DEFAULT_FLAG_VALUES

    hacked = conc.copy()
    hacked[hacked == -9999] = DEFAULT_FLAG_VALUES.land
    hacked[hacked == -9998] = DEFAULT_FLAG_VALUES.coast
    hacked[hacked == -50] = DEFAULT_FLAG_VALUES.pole_hole
    hacked[hacked == -10] = DEFAULT_FLAG_VALUES.missing

    return hacked


def test_nt_f17_regression_north():
    """Regression test for NT F17 output."""
    regression_data = np.fromfile(
        REGRESSION_DATA_DIR / 'nt_f17_regression' / 'NH_f17_20180101_int16.dat',
        dtype=np.int16,
    ).reshape(get_ps25_grid_shape(hemisphere='north'))

    actual_ds = original_example(hemisphere='north')

    hacked = _hack_flag_vals(regression_data)

    assert_equal(
        hacked,
        actual_ds.conc.data,
    )


def test_nt_f17_regression_south():
    """Regression test for NT F17 output."""
    actual_ds = original_example(hemisphere='south')

    regression_data = np.fromfile(
        REGRESSION_DATA_DIR / 'nt_f17_regression' / 'SH_f17_20180101_int16.dat',
        dtype=np.int16,
    ).reshape(get_ps25_grid_shape(hemisphere='south'))

    hacked = _hack_flag_vals(regression_data)

    assert_equal(
        hacked,
        actual_ds.conc.data,
    )
