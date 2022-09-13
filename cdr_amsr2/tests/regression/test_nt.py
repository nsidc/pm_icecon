from typing import get_args

import numpy as np
from numpy.testing import assert_almost_equal

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

        # Scale down by 10
        regression_data = regression_data / 10  # type: ignore

        # Clamp concentrations to a max of 100
        regression_data[(regression_data > 100) & (regression_data < 200)] = 100

        # assert that the only differences are land values. For some reason
        # (TODO - investigate), the regression data does not have the southern
        # tip of south america in the data.
        if hemisphere == 'south':
            diff = np.abs(regression_data - actual_ds.conc.data)
            meaningful_diff = diff > 0.1
            assert np.all(diff[meaningful_diff] == 254)
        else:
            assert_almost_equal(regression_data, actual_ds.conc.data, decimal=1)
