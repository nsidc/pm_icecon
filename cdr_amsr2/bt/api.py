import datetime as dt
from pathlib import Path

import xarray as xr

from cdr_amsr2._types import Hemisphere
from cdr_amsr2.constants import PACKAGE_DIR
import cdr_amsr2.bt.compute_bt_ic as bt
from cdr_amsr2.fetch.au_si25 import get_au_si25_tbs


def amsr2_bootstrap(*, date: dt.date, hemisphere: Hemisphere) -> xr.Dataset:
    """Compute sea ice concentration from AU_SI25 TBs."""
    if hemisphere == 'south':
        raise NotImplementedError(
            'Southern hemisphere is not currently supported.'
        )

    xr_tbs = get_au_si25_tbs(
        base_dir=Path('/ecs/DP1/AMSA/AU_SI25.001/'),
        date=date,
        hemisphere='north',
    )

    params = bt.import_cfg_file(PACKAGE_DIR / 'bt' / 'ret_ic_params_amsru.json')
    variables = bt.import_cfg_file(PACKAGE_DIR / 'bt' / 'ret_ic_variables_amsru.json')

    tbs = {
        'v19': xr_tbs['v18'].data,
        'v37': xr_tbs['v36'].data,
        'h37': xr_tbs['h36'].data,
        'v22': xr_tbs['v23'].data,
    }

    conc_ds = bt.bootstrap(
        tbs=tbs,
        params=params,
        variables=variables,
    )

    return conc_ds
