from typing import get_args

import xarray as xr
from numpy.testing import assert_almost_equal

from cdr_amsr2._types import Hemisphere
from cdr_amsr2.constants import CDR_TESTDATA_DIR
from cdr_amsr2.nt.api import original_example


def test_nt_f17_regressions():
    """Regression test for NT F17 output."""
    for hemisphere in get_args(Hemisphere):

        regression_ds = xr.open_dataset(
            CDR_TESTDATA_DIR
            / 'nt_f17_regression'
            / f'{hemisphere[0].upper()}H_f17_20180101_regression.nc',
        )
        regression_data = regression_ds.conc.data

        actual_ds = original_example(hemisphere=hemisphere)
        actual_data = actual_ds.conc.data

        assert_almost_equal(
            regression_data,
            actual_data,
            decimal=1,
        )
