import datetime as dt

import numpy as np
import xarray as xr
from numpy.testing import assert_equal

from cdr_amsr2.bt.api import amsr2_bootstrap, original_f18_example
from cdr_amsr2.tests.regression.util import REGRESSION_DATA_DIR


def _hack_flag_vals(conc):
    from cdr_amsr2.constants import DEFAULT_FLAG_VALUES

    hacked = conc.copy()
    hacked[hacked == 1200] = DEFAULT_FLAG_VALUES.land
    hacked[hacked == 1100] = DEFAULT_FLAG_VALUES.missing

    return hacked


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
            resolution='25',
        )

        # HACK: make the regression data use the new flag values.
        hacked = _hack_flag_vals(regression_ds.conc.data)
        not_eq = hacked != actual_ds.conc.data
        assert np.all(actual_ds.conc.data[not_eq] == 0)

        # assert_equal(
        #     hacked,
        #     actual_ds.conc.data,
        # )


def test_bt_f18_regression():
    """Regressi5on test for BT F18 output."""
    regression_data = np.fromfile(
        REGRESSION_DATA_DIR / 'bt_f18_regression/NH_20180217_py_NRT_f18.ic',
        dtype=np.int16,
    ).reshape((448, 304))

    actual_ds = original_f18_example()

    hacked = _hack_flag_vals(regression_data)

    assert_equal(
        hacked,
        actual_ds.conc.data,
    )
