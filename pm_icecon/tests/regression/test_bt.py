import datetime as dt
from pathlib import Path
from typing import Final

import numpy as np
import xarray as xr
from numpy.testing import assert_almost_equal
from numpy.typing import NDArray
import numpy.typing as npt
from pm_tb_data.fetch.au_si import AU_SI_RESOLUTIONS
from pm_tb_data._types import Hemisphere

import pm_icecon.bt.compute_bt_ic as bt
from pm_icecon.bt.api import amsr2_goddard_bootstrap
from pm_icecon.bt.compute_bt_ic import xfer_class_tbs
from pm_icecon.bt.masks import (
    get_ps_invalid_ice_mask,
    get_pss_12_validice_land_coast_array,
)
from pm_icecon.bt.params.class_sats import SSMIS_NORTH_PARAMS
from pm_icecon.config.models.bt import BootstrapParams
from pm_icecon.interpolation import spatial_interp_tbs
from pm_icecon.util import get_ps25_grid_shape
from pm_icecon.constants import (
    BT_GODDARD_ANCILLARY_DIR,
    CDR_TESTDATA_DIR,
)


def _get_ps_land_mask(
    *,
    hemisphere: Hemisphere,
    resolution: AU_SI_RESOLUTIONS,
) -> npt.NDArray[np.bool_]:
    """Get the polar stereo 25km land mask."""
    # Ocean has a value of 0, land a value of 1, and coast a value of 2.
    if resolution == "25":
        shape = get_ps25_grid_shape(hemisphere=hemisphere)
        _land_coast_array = np.fromfile(
            (
                BT_GODDARD_ANCILLARY_DIR
                / (
                    f"{hemisphere}_land_25"
                    # NOTE: According to scotts, the 'r' in the southern hemisphere
                    # filename probably stands for “revised“.
                    f"{'r' if hemisphere == 'south' else ''}"
                )
            ).resolve(),
            dtype=np.int16,
        ).reshape(shape)

        land_mask = _land_coast_array != 0

        # TODO: land mask currently includes land and coast. Does this make sense? Are
        # we ever going to need to coast values? Maybe rename to `LAND_COAST_MASK`?
    elif resolution == "12":
        if hemisphere == "south":
            # Any date is OK. The land mask is the same for all of the pss 12
            # validice/land masks
            _land_coast_array = get_pss_12_validice_land_coast_array(
                date=dt.date.today()
            )
            land_mask = np.logical_or(_land_coast_array == 0, _land_coast_array == 32)
        else:
            _land_coast_array = np.fromfile(
                CDR_TESTDATA_DIR / "btequiv_psn12.5/bt_landequiv_psn12.5km.dat",
                dtype=np.int16,
            ).reshape(896, 608)

            land_mask = _land_coast_array != 0

    return land_mask


def test_bt_amsr2_regression():
    """Regression test for BT AMSR2 outputs.

    Compare output from bt algorithm for 2020-01-01 and 2022-05-04 against
    regression data.

    Scott Stewart manually examined the regression data and determined it looks
    good. These fields may need to be updated as we make tweaks to the
    algorithm.
    """
    resolution: Final = "25"
    hemisphere: Final = "north"

    land_mask = _get_ps_land_mask(hemisphere=hemisphere, resolution=resolution)
    for date in (dt.date(2020, 1, 1), dt.date(2022, 5, 4)):
        invalid_ice_mask = get_ps_invalid_ice_mask(
            hemisphere=hemisphere,
            date=date,
            resolution=resolution,
        )
        actual_ds = amsr2_goddard_bootstrap(
            date=date,
            hemisphere=hemisphere,
            resolution=resolution,
            land_mask=land_mask,
            invalid_ice_mask=invalid_ice_mask,
        )

        filename = f"NH_{date:%Y%m%d}_py_NRT_amsr2.nc"
        regression_ds = xr.open_dataset(
            CDR_TESTDATA_DIR / "bt_amsru_regression" / filename
        )

        assert_almost_equal(
            regression_ds.conc.data,
            actual_ds.conc.data,
            decimal=3,
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
    resolution: AU_SI_RESOLUTIONS = "25"
    date = dt.date(2018, 2, 17)
    hemisphere: Hemisphere = "north"

    land_mask = _get_ps_land_mask(hemisphere=hemisphere, resolution=resolution)
    invalid_ice_mask = get_ps_invalid_ice_mask(
        hemisphere=hemisphere,
        date=date,
        resolution=resolution,
    )
    params = BootstrapParams(
        **SSMIS_NORTH_PARAMS,  # type: ignore[arg-type]
    )

    otbs: dict[str, NDArray[np.float32]] = {}

    orig_input_tbs_dir = CDR_TESTDATA_DIR / "bt_goddard_orig_input_tbs/"
    raw_fns = {
        "v19": "tb_f18_20180217_nrt_n19v.bin",
        "h37": "tb_f18_20180217_nrt_n37h.bin",
        "v37": "tb_f18_20180217_nrt_n37v.bin",
        "v22": "tb_f18_20180217_nrt_n22v.bin",
    }

    def _read_tb_field(tbfn: Path) -> NDArray[np.float32]:
        # Read int16 scaled by 10 and return float32 unscaled
        raw = np.fromfile(tbfn, dtype=np.int16).reshape(448, 304)

        return raw.astype(np.float32) / 10

    for tb in ("v19", "h37", "v37", "v22"):
        otbs[tb] = _read_tb_field((orig_input_tbs_dir / raw_fns[tb]).resolve())

    conc_ds = bt.goddard_bootstrap(
        # Apply expected transformation for F18 CLASS data.
        **xfer_class_tbs(  # type: ignore[arg-type]
            tb_v37=spatial_interp_tbs(otbs["v37"]),
            tb_h37=spatial_interp_tbs(otbs["h37"]),
            tb_v19=spatial_interp_tbs(otbs["v19"]),
            tb_v22=spatial_interp_tbs(otbs["v22"]),
            sat="f18",
        ),
        params=params,
        date=date,
        land_mask=land_mask,
        invalid_ice_mask=invalid_ice_mask,
    )

    return conc_ds


def test_bt_f18_regression():
    """Regressi5on test for BT F18 output."""
    actual_ds = _original_f18_example()
    regression_ds = xr.open_dataset(
        CDR_TESTDATA_DIR / "bt_f18_regression/NH_20180217_NRT_f18_regression.nc",
    )

    assert_almost_equal(
        regression_ds.conc.data,
        actual_ds.conc.data,
        decimal=3,
    )
