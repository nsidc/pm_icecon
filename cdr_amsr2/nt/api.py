from pathlib import Path

import numpy as np
import xarray as xr

from cdr_amsr2._types import Hemisphere
from cdr_amsr2.constants import PACKAGE_DIR
from cdr_amsr2.nt.compute_nt_ic import nasateam
from cdr_amsr2.util import get_ps25_grid_shape


def original_example(*, hemisphere: Hemisphere) -> xr.Dataset:
    """Return the concentration field example for f17_20180101."""
    raw_fns = {
        'h19': f'tb_f17_20180101_v4_{hemisphere[0].lower()}19h.bin',
        'v19': f'tb_f17_20180101_v4_{hemisphere[0].lower()}19v.bin',
        'v22': f'tb_f17_20180101_v4_{hemisphere[0].lower()}22v.bin',
        'h37': f'tb_f17_20180101_v4_{hemisphere[0].lower()}37h.bin',
        'v37': f'tb_f17_20180101_v4_{hemisphere[0].lower()}37v.bin',
    }

    tbs = {}
    for tb in raw_fns.keys():
        tbfn = raw_fns[tb]
        grid_shape = get_ps25_grid_shape(hemisphere=hemisphere)
        tbs[tb] = np.fromfile(
            Path('/share/apps/amsr2-cdr/cdr_testdata/nt_goddard_input_tbs/') / tbfn,
            dtype=np.int16,
        ).reshape(grid_shape)

    shoremap_fn = (
        PACKAGE_DIR
        / '..'
        / f'legacy/nt_orig/DATAFILES/data36/maps/shoremap_{hemisphere}_25'
    )
    shoremap = np.fromfile(shoremap_fn, dtype='>i2')[150:].reshape(grid_shape)

    # TODO: why is 'SSMI8' on FH fn and not SH?
    if hemisphere == 'north':
        minic_fn = 'SSMI8_monavg_min_con'
    else:
        minic_fn = 'SSMI_monavg_min_con_s'

    minic_path = PACKAGE_DIR / '..' / 'legacy/nt_orig/DATAFILES/data36/maps' / minic_fn
    minic = np.fromfile(minic_path, dtype='>i2')[150:].reshape(grid_shape)

    conc_ds = nasateam(
        tbs=tbs,
        sat='17_final',
        hemisphere=hemisphere,
        shoremap=shoremap,
        minic=minic,
    )

    return conc_ds
