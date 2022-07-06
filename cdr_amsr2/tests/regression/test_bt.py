from pathlib import Path

from numpy.testing import assert_equal
import xarray as xr

from cdr_amsr2.constants import PACKAGE_DIR


REGRESSION_DATA_DIR = Path('/share/apps/amsr2-cdr/cdr_testdata/bt_amsru_regression')


def test_bt_amsr2_regression():
    """Compare output from bt algorithm for 2020-01-01 and 2022-05-04 against
    regression data.

    Scott Stewart manually examined the regression data and determined it looks
    good. These fields may need to be updated as we make tweaks to the
    algorithm.
    """
    for date_str in ('20200101', '20220504'):
        filename = f'NH_{date_str}_py_NRT_amsr2.nc'
        regression_ds = xr.open_dataset(REGRESSION_DATA_DIR / filename)
        actual_ds = xr.open_dataset((PACKAGE_DIR / '..' / filename).resolve())

        assert_equal(
            regression_ds.conc.data,
            actual_ds.conc.data,
        )
