from typing import get_args

import numpy as np

from cdr_amsr2._types import Hemisphere
from cdr_amsr2.constants import DEFAULT_FLAG_VALUES
from cdr_amsr2.nt.api import original_example
from cdr_amsr2.tests.regression.util import REGRESSION_DATA_DIR
from cdr_amsr2.util import get_ps25_grid_shape

# from numpy.testing import assert_equal


def _hack_flag_vals(conc):

    hacked = conc.copy()
    hacked[hacked == -9999] = DEFAULT_FLAG_VALUES.land
    # make coastlines into 'land' for now.
    hacked[hacked == -9998] = DEFAULT_FLAG_VALUES.land
    hacked[hacked == -50] = DEFAULT_FLAG_VALUES.pole_hole
    hacked[hacked == -10] = 0

    return hacked


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

        hacked = _hack_flag_vals(regression_data)

        # after removing the pole hole flag from nasateam, this should be the
        # only difference.
        not_eq = hacked != actual_ds.conc.data
        assert np.all(hacked[not_eq] == DEFAULT_FLAG_VALUES.pole_hole)

        # assert_equal(
        #     regression_data,
        #     actual_ds.conc.data,
        # )
