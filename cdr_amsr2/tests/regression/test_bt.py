import datetime as dt
from pathlib import Path

import numpy as np
import xarray as xr
from numpy.testing import assert_equal

from cdr_amsr2.bt.api import amsr2_bootstrap, original_f18_example

REGRESSION_DATA_DIR = Path('/share/apps/amsr2-cdr/cdr_testdata')


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
        regression_ds = xr.open_dataset(
            REGRESSION_DATA_DIR / 'bt_amsru_regression' / filename
        )
        actual_ds = amsr2_bootstrap(
            date=date,
            hemisphere='north',
        )

        assert_equal(
            regression_ds.conc.data,
            actual_ds.conc.data,
        )


def test_bt_f18_regression():
    """Regressi5on test for BT F18 output."""
    regression_data = np.fromfile(
        REGRESSION_DATA_DIR / 'bt_f18_regression/NH_20180217_py_NRT_f18.ic',
        dtype=np.int16,
    ).reshape((448, 304))

    actual_ds = original_f18_example()

    assert_equal(
        regression_data,
        actual_ds.conc.data,
    )
