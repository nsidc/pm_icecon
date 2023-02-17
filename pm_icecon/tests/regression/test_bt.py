import datetime as dt
from pathlib import Path

import numpy as np
import xarray as xr
from numpy.testing import assert_almost_equal
from numpy.typing import NDArray

import pm_icecon.bt.compute_bt_ic as bt
from pm_icecon._types import Hemisphere
from pm_icecon.bt.api import amsr2_bootstrap
from pm_icecon.bt.compute_bt_ic import xfer_class_tbs
from pm_icecon.bt.masks import get_ps_invalid_ice_mask
from pm_icecon.bt.params.goddard_class import SSMIS_NORTH_PARAMS
from pm_icecon.config.models.bt import BootstrapParams
from pm_icecon.constants import CDR_TESTDATA_DIR
from pm_icecon.fetch.au_si import AU_SI_RESOLUTIONS
from pm_icecon.interpolation import spatial_interp_tbs
from pm_icecon.masks import get_ps_land_mask, get_ps_pole_hole_mask


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
            CDR_TESTDATA_DIR / 'bt_amsru_regression' / filename
        )
        assert_almost_equal(
            regression_ds.conc.data,
            actual_ds.conc.data,
            decimal=1,
        )


def _original_f18_example() -> xr.Dataset:
    """Return concentration field example for f18_20180217.

    This example data does not perfectly match the outputs given by Goddard's
    code, but it is very close. A total of 4 cells differ 1.

    ```
    >>> exact[not_eq]
    array([984, 991, 975, 830], dtype=int16)
    >>> not_eq = exact != not_exact
    >>> not_exact[not_eq]
    array([983, 992, 974, 829], dtype=int16)
    ```

    the exact grid produced by the fortran code is in
    `CDR_TESTDATA / 'bt_goddard_orig_output/NH_20180217_SB2_NRT_f18.ic'`.
    """
    resolution: AU_SI_RESOLUTIONS = '25'
    date = dt.date(2018, 2, 17)
    hemisphere: Hemisphere = 'north'
    params = BootstrapParams(
        land_mask=get_ps_land_mask(hemisphere=hemisphere, resolution=resolution),
        pole_mask=get_ps_pole_hole_mask(resolution=resolution),
        invalid_ice_mask=get_ps_invalid_ice_mask(
            hemisphere=hemisphere,
            date=date,
            resolution=resolution,  # type: ignore[arg-type]
        ),
        **SSMIS_NORTH_PARAMS,
    )

    otbs: dict[str, NDArray[np.float32]] = {}

    orig_input_tbs_dir = CDR_TESTDATA_DIR / 'bt_goddard_orig_input_tbs/'
    raw_fns = {
        'v19': 'tb_f18_20180217_nrt_n19v.bin',
        'h37': 'tb_f18_20180217_nrt_n37h.bin',
        'v37': 'tb_f18_20180217_nrt_n37v.bin',
        'v22': 'tb_f18_20180217_nrt_n22v.bin',
    }

    def _read_tb_field(tbfn: Path) -> NDArray[np.float32]:
        # Read int16 scaled by 10 and return float32 unscaled
        raw = np.fromfile(tbfn, dtype=np.int16).reshape(448, 304)

        return bt.fdiv(raw.astype(np.float32), 10)

    for tb in ('v19', 'h37', 'v37', 'v22'):
        otbs[tb] = _read_tb_field(
            (
                orig_input_tbs_dir / raw_fns[tb]  # type: ignore [literal-required]
            ).resolve()
        )

    conc_ds = bt.goddard_bootstrap(
        # Apply expected transformation for F18 CLASS data.
        **xfer_class_tbs(  # type: ignore[arg-type]
            tb_v37=spatial_interp_tbs(otbs['v37']),
            tb_h37=spatial_interp_tbs(otbs['h37']),
            tb_v19=spatial_interp_tbs(otbs['v19']),
            tb_v22=spatial_interp_tbs(otbs['v22']),
            sat='f18',
        ),
        params=params,
        date=date,
        hemisphere=hemisphere,
    )

    return conc_ds


def test_bt_f18_regression():
    """Regressi5on test for BT F18 output."""
    actual_ds = _original_f18_example()
    regression_ds = xr.open_dataset(
        CDR_TESTDATA_DIR / 'bt_f18_regression/NH_20180217_NRT_f18_regression.nc',
    )

    assert_almost_equal(
        regression_ds.conc.data,
        actual_ds.conc.data,
        decimal=1,
    )
