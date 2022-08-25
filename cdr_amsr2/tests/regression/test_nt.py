from pathlib import Path

import numpy as np
from numpy.testing import assert_equal

from cdr_amsr2.util import get_ps25_grid_shape
from cdr_amsr2.nt.api import original_example
from cdr_amsr2.tests.regression.util import REGRESSION_DATA_DIR


def _read_goddard_nasateam_file(filename: Path, /) -> np.ndarray:
    with open(filename, 'rb') as fp:
        fp.read(300)
        data = np.fromfile(fp, dtype='>i2')
    return data


def test_nt_f17_regression_north():
    """Regression test for NT F17 output."""
    regression_data = _read_goddard_nasateam_file(
        REGRESSION_DATA_DIR / 'nt_f17_regression' / 'nssss1d17tcon2018001.spill_sst',
    ).reshape(get_ps25_grid_shape(hemisphere='north'))

    actual_ds = original_example(hemisphere='north')

    # min/max are the same, but 10 pixels differ
    # ipdb> not_eq = actual_ds.conc.data != regression_data
    # ipdb> actual_data[not_eq]
    # array([  0, 161, 280, 138, 166,  80, 261, 242, 111,  79], dtype=int16)
    # ipdb> regression_data[not_eq]
    # array([301, 187, 322, 393, 218, 593, 369, 253, 396, 571], dtype=int16)

    not_eq = actual_ds.conc.data != regression_data
    print(f'Number of items not equal: {np.sum(not_eq)}')
    # 10 differences (with do_exact=True).
    breakpoint()

    assert_equal(
        regression_data,
        actual_ds.conc.data,
    )


def test_nt_f17_regression_south():
    """Regression test for NT F17 output."""
    actual_ds = original_example(hemisphere='south')

    regression_data = _read_goddard_nasateam_file(
        REGRESSION_DATA_DIR / 'nt_f17_regression' / 'sssss1d17tcon2018001.spill_sst'
    ).reshape(get_ps25_grid_shape(hemisphere='south'))

    not_eq = actual_ds.conc.data != regression_data
    print(f'Number of items not equal: {np.sum(not_eq)}')
    # 129 differences.

    assert_equal(
        regression_data,
        actual_ds.conc.data,
    )
