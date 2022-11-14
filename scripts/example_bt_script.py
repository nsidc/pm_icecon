"""Example python script for running bootstrap with a2l1c data.

NOTE: this is for demonstration purposes only and should not be run without
modifications.
"""
import datetime as dt
from pathlib import Path

import numpy as np

from cdr_amsr2._types import Hemisphere
from cdr_amsr2.bt.compute_bt_ic import bootstrap
from cdr_amsr2.bt.params.a2l1c import A2L1C_NORTH_PARAMS
from cdr_amsr2.config.models.bt import BootstrapParams
from cdr_amsr2.fetch.a2l1c_625 import get_a2l1c_625_tbs

if __name__ == '__main__':
    hemisphere: Hemisphere = 'north'
    date = dt.date(2018, 8, 1)

    # Get the brightness temperatures for a2l1c
    # NOTE/TODO: change the `base_dir` path here to your local data location!
    xr_tbs = get_a2l1c_625_tbs(
        base_dir=Path('/path/to/a2l1c_data_location/'),
        date=date,
        hemisphere=hemisphere,
    )

    # Define required masks
    # NOTE/TODO: replace these with 'real' masks!
    _data_shape = xr_tbs['v18'].shape
    land_mask = np.zeros(_data_shape).astype(bool)
    pole_mask = np.zeros(_data_shape).astype(bool)
    invalid_ice_mask = np.zeros(_data_shape).astype(bool)

    # Define bootstrap params. See
    # `cdr_amsr2/config/models/bt.py:BootstrapParams` for more details.
    bootstrap_params = BootstrapParams(
        land_mask=land_mask,
        pole_mask=pole_mask,
        invalid_ice_mask=invalid_ice_mask,
        **A2L1C_NORTH_PARAMS,
    )

    # Run the bootstrap algoirthm and get the result back as an xarray dataset.
    conc_ds = bootstrap(
        tb_v37=xr_tbs['v36'].data,
        tb_h37=xr_tbs['h36'].data,
        tb_v19=xr_tbs['v18'].data,
        tb_v22=xr_tbs['v23'].data,
        params=bootstrap_params,
        date=date,
        hemisphere=hemisphere,
    )
