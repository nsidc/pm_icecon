import datetime as dt

import numpy as np
import xarray as xr
from numpy.testing import assert_almost_equal

from cdr_amsr2.bt.api import amsr2_bootstrap, original_f18_example
from cdr_amsr2.tests.regression.util import REGRESSION_DATA_DIR


def test_bt_amsr2_regression():
    """Regression test for BT AMSR2 outputs.

    Compare output from bt algorithm for 2020-01-01 and 2022-05-04 against
    regression data.

    Scott Stewart manually examined the regression data and determined it looks
    good. These fields may need to be updated as we make tweaks to the
    algorithm.
    """
    for date in (dt.date(2020, 1, 1), dt.date(2022, 5, 4)):
        actual_ds = amsr2_bootstrap(
            date=date,
            hemisphere='north',
            resolution='25',
        )
        filename = f'NH_{date:%Y%m%d}_py_NRT_amsr2.nc'
        regression_ds = xr.open_dataset(
            REGRESSION_DATA_DIR / 'bt_amsru_regression' / filename
        )
        try:
            assert_almost_equal(
                regression_ds.conc.data,
                actual_ds.conc.data,
                decimal=1,
            )
        except AssertionError as e:
            print('Failed test_bt_amsr2_regression')
            print(f'  date: {date}')
            print(f'  arg to xr: {REGRESSION_DATA_DIR / "bt_amsru_regression" / filename}')  # noqa
            print(f'Type of actual_ds: {type(actual_ds)}')
            # ofn = f'replacement_ds_{date}.nc'
            # actual_ds.to_netcdf(ofn)
            # print(f'  Wrote "actual_ds" to: {ofn}')
            raise e


def test_bt_f18_regression():
    """Regressi5on test for BT F18 output."""
    actual_ds = original_f18_example()
    """
    regression_data = np.fromfile(
        REGRESSION_DATA_DIR / 'bt_f18_regression/NH_20180217_py_NRT_f18.ic',
        dtype=np.int16,
    ).reshape((448, 304))
    """
    regression_ds = xr.open_dataset(
        REGRESSION_DATA_DIR / 'bt_f18_regression/NH_20180217_NRT_f18_regression.nc',
    )

    try:
        assert_almost_equal(
            # regression_data / 10,
            regression_ds.conc.data,
            actual_ds.conc.data,
            decimal=1,
        )
    except AssertionError as e:
        print('Failed test_bt_f18_regression')
        print(f'  arg to xr: {REGRESSION_DATA_DIR / "bt_f18_regression/NH_20180217_py_NRT_f18.ic"}')  # noqa
        print(f'Type of actual_ds: {type(actual_ds)}')
        # ofn = f'replacement_ds_for_bt_f18.nc'
        # actual_ds.to_netcdf(ofn)
        # print(f'  Wrote "actual_ds" to: {ofn}')
        print(f'{e}')
        raise e
