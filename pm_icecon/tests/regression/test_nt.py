import datetime as dt
from typing import get_args

import numpy as np
import xarray as xr
from numpy.testing import assert_almost_equal

from pm_icecon._types import Hemisphere
from pm_icecon.constants import CDR_TESTDATA_DIR
from pm_icecon.interpolation import spatial_interp_tbs
from pm_icecon.nt.compute_nt_ic import nasateam
from pm_icecon.nt.masks import get_ps25_sst_mask
from pm_icecon.nt.params.goddard_rss import (
    RSS_F17_NORTH_GRADIENT_THRESHOLDS,
    RSS_F17_SOUTH_GRADIENT_THRESHOLDS,
)
from pm_icecon.nt.tiepoints import get_tiepoints
from pm_icecon.util import get_ps25_grid_shape


def _original_example(*, hemisphere: Hemisphere) -> xr.Dataset:
    """Return the concentration field example for f17_20180101."""
    _nt_maps_dir = CDR_TESTDATA_DIR / 'nt_datafiles/data36/maps/'

    def _get_shoremap(*, hemisphere: Hemisphere):
        shoremap_fn = _nt_maps_dir / f'shoremap_{hemisphere}_25'
        shoremap = np.fromfile(shoremap_fn, dtype='>i2')[150:].reshape(
            get_ps25_grid_shape(hemisphere=hemisphere)
        )

        return shoremap

    def _get_minic(*, hemisphere: Hemisphere):
        # TODO: why is 'SSMI8' on FH fn and not SH?
        if hemisphere == 'north':
            minic_fn = 'SSMI8_monavg_min_con'
        else:
            minic_fn = 'SSMI_monavg_min_con_s'

        minic_path = _nt_maps_dir / minic_fn
        minic = np.fromfile(minic_path, dtype='>i2')[150:].reshape(
            get_ps25_grid_shape(hemisphere=hemisphere)
        )

        # Scale down by 10. The original alg. dealt w/ concentrations scaled by 10.
        minic = minic / 10

        return minic

    date = dt.date(2018, 1, 1)
    orig_input_tbs_dir = CDR_TESTDATA_DIR / 'nt_goddard_input_tbs'
    raw_fns = {
        'v19': f'tb_f17_{date:%Y%m%d}_v4_{hemisphere[0].lower()}19v.bin',
        'v37': f'tb_f17_{date:%Y%m%d}_v4_{hemisphere[0].lower()}37v.bin',
        'v22': f'tb_f17_{date:%Y%m%d}_v4_{hemisphere[0].lower()}22v.bin',
        'h19': f'tb_f17_{date:%Y%m%d}_v4_{hemisphere[0].lower()}19h.bin',
    }

    tbs = {}
    grid_shape = get_ps25_grid_shape(hemisphere=hemisphere)
    for tb in raw_fns.keys():
        tbfn = raw_fns[tb]
        tbs[tb] = spatial_interp_tbs(
            np.fromfile(
                orig_input_tbs_dir / tbfn,
                dtype=np.int16,
            ).reshape(grid_shape)
        )

    invalid_ice_mask = get_ps25_sst_mask(hemisphere=hemisphere, date=date)

    conc_ds = nasateam(
        tb_v19=tbs['v19'],
        tb_v37=tbs['v37'],
        tb_v22=tbs['v22'],
        tb_h19=tbs['h19'],
        hemisphere=hemisphere,
        shoremap=_get_shoremap(hemisphere=hemisphere),
        minic=_get_minic(hemisphere=hemisphere),
        date=date,
        invalid_ice_mask=invalid_ice_mask,
        gradient_thresholds=(
            RSS_F17_NORTH_GRADIENT_THRESHOLDS
            if hemisphere == 'north'
            else RSS_F17_SOUTH_GRADIENT_THRESHOLDS
        ),
        tiepoints=get_tiepoints(satellite='17_final', hemisphere=hemisphere),
    )

    return conc_ds


def test_nt_f17_regressions():
    """Regression test for NT F17 output."""
    for hemisphere in get_args(Hemisphere):

        regression_ds = xr.open_dataset(
            CDR_TESTDATA_DIR
            / 'nt_f17_regression'
            / f'{hemisphere[0].upper()}H_f17_20180101_regression.nc',
        )
        regression_data = regression_ds.conc.data

        actual_ds = _original_example(hemisphere=hemisphere)
        actual_data = actual_ds.conc.data

        assert_almost_equal(
            regression_data,
            actual_data,
            decimal=1,
        )
