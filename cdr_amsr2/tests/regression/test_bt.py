import datetime as dt
from pathlib import Path

import xarray as xr
from numpy.testing import assert_equal

from cdr_amsr2.bt.api import amsr2_bootstrap

REGRESSION_DATA_DIR = Path('/share/apps/amsr2-cdr/cdr_testdata/bt_amsru_regression')


def test_bt_amsr2_regression():
    """Regression test for BT AMSR2 outputs.

    Compare output from bt algorithm for 2020-01-01 and 2022-05-04 against
    regression data.

    Scott Stewart manually examined the regression data and determined it looks
    good. These fields may need to be updated as we make tweaks to the
    algorithm.
    """
    for date in (dt.date(2020, 1, 1), dt.date(2022, 5, 4)):
        filename = f'NH_{date:%Y%m%d}_py_NRT_amsr2.nc'
        regression_ds = xr.open_dataset(REGRESSION_DATA_DIR / filename)
        actual_ds = amsr2_bootstrap(
            date=date,
            hemisphere='north',
        )

        assert_equal(
            regression_ds.conc.data,
            actual_ds.conc.data,
        )
