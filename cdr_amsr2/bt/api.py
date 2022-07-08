import datetime as dt
from pathlib import Path

import numpy as np
import numpy.typing as npt
import xarray as xr

import cdr_amsr2.bt.compute_bt_ic as bt
from cdr_amsr2._types import Hemisphere
from cdr_amsr2.config import import_cfg_file
from cdr_amsr2.constants import PACKAGE_DIR
from cdr_amsr2.fetch.au_si25 import get_au_si25_tbs


def amsr2_bootstrap(*, date: dt.date, hemisphere: Hemisphere) -> xr.Dataset:
    """Compute sea ice concentration from AU_SI25 TBs."""
    if hemisphere == 'south':
        raise NotImplementedError('Southern hemisphere is not currently supported.')

    xr_tbs = get_au_si25_tbs(
        base_dir=Path('/ecs/DP1/AMSA/AU_SI25.001/'),
        date=date,
        hemisphere='north',
    )

    params = import_cfg_file(PACKAGE_DIR / 'bt' / 'ret_ic_params_amsru.json')
    variables = import_cfg_file(PACKAGE_DIR / 'bt' / 'ret_ic_variables_amsru.json')

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


def original_f18_example() -> xr.Dataset:
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
    `legacy/SB2_NRT_programs/NH_20180217_SB2_NRT_f18.ic`
    """
    params = import_cfg_file(PACKAGE_DIR / 'bt' / 'ret_ic_params.json')
    variables = import_cfg_file(PACKAGE_DIR / 'bt' / 'ret_ic_variables.json')

    otbs: dict[str, npt.NDArray[np.float32]] = {}

    for tb in ('v19', 'h37', 'v37', 'v22'):
        otbs[tb] = bt.read_tb_field(
            (
                PACKAGE_DIR
                / '../legacy/SB2_NRT_programs'
                / params['raw_fns'][tb]  # type: ignore [literal-required]
            ).resolve()
        )

    conc_ds = bt.bootstrap(
        tbs=otbs,
        params=params,
        variables=variables,
    )

    return conc_ds
