import numpy as np
import xarray as xr
from xarray.testing import assert_equal

from cdr_amsr2.fetch import au_si25


def test__normalize_au_si25_tbs():
    mock_au_si25_data_fields = xr.Dataset(
        data_vars={
            'SI_25km_NH_06H_DAY': ('x', np.arange(0, 5)),
            'SI_25km_NH_89V_DAY': ('x', np.arange(5, 10)),
        },
    )

    expected = xr.Dataset(
        data_vars={
            'h06': ('x', np.arange(0, 5)),
            'v89': ('x', np.arange(5, 10)),
        },
    )
    actual = au_si25._normalize_au_si25_tbs(
        mock_au_si25_data_fields,
    )

    assert_equal(actual, expected)
