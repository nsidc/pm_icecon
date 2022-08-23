import numpy as np
import xarray as xr

from cdr_amsr2.config import import_cfg_file
from cdr_amsr2.constants import PACKAGE_DIR
from cdr_amsr2.nt.compute_nt_ic import nasateam


def original_example() -> xr.Dataset:
    """Return the concentration field example for f17_20180101."""
    params = import_cfg_file(PACKAGE_DIR / 'nt' / 'nt_sample_nh.json')

    tbs = {}
    for tb in ('v19', 'h19', 'v22', 'h37', 'v37'):
        tbfn = params['raw_fns'][tb]
        tbs[tb] = np.fromfile(tbfn, dtype=np.int16).reshape(448, 304)

    conc_ds = nasateam(tbs=tbs, sat='17', hemisphere='north')

    return conc_ds
