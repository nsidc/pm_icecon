from typing import get_args

import numpy as np
from numpy.testing import assert_equal

from cdr_amsr2._types import Hemisphere
from cdr_amsr2.nt.api import original_example
from cdr_amsr2.tests.regression.util import REGRESSION_DATA_DIR
from cdr_amsr2.util import get_ps25_grid_shape


def test_nt_f17_regressions():
    """Regression test for NT F17 output."""
    for hemisphere in get_args(Hemisphere):
        regression_data = np.fromfile(
            REGRESSION_DATA_DIR
            / 'nt_f17_regression'
            / f'{hemisphere[0].upper()}H_f17_20180101_int16.dat',
            dtype=np.int16,
        ).reshape(get_ps25_grid_shape(hemisphere=hemisphere))

        actual_ds = original_example(hemisphere=hemisphere)

        assert_equal(
            regression_data,
            actual_ds.conc.data,
        )
