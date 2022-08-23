from pathlib import Path

import numpy as np
import xarray as xr

from cdr_amsr2.nt.compute_nt_ic import nasateam


def original_example() -> xr.Dataset:
    """Return the concentration field example for f17_20180101."""
    raw_fns = {
        'h19': 'tb_f17_20180101_v4_n19h.bin',
        'v19': 'tb_f17_20180101_v4_n19v.bin',
        'v22': 'tb_f17_20180101_v4_n22v.bin',
        'h37': 'tb_f17_20180101_v4_n37h.bin',
        'v37': 'tb_f17_20180101_v4_n37v.bin',
    }

    tbs = {}
    for tb in raw_fns.keys():
        tbfn = raw_fns[tb]
        tbs[tb] = np.fromfile(
            Path('/share/apps/amsr2-cdr/cdr_testdata/nt_goddard_input_tbs/') / tbfn,
            dtype=np.int16,
        ).reshape(448, 304)

    conc_ds = nasateam(tbs=tbs, sat='17', hemisphere='north')

    return conc_ds
