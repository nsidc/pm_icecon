import datetime as dt
from pathlib import Path

import numpy as np
import xarray as xr

from cdr_amsr2._types import Hemisphere
from cdr_amsr2.constants import PACKAGE_DIR
from cdr_amsr2.fetch.au_si import get_au_si_tbs
from cdr_amsr2.nt.compute_nt_ic import nasateam
from cdr_amsr2.util import get_ps25_grid_shape


def _get_shoremap(*, hemisphere: Hemisphere):
    shoremap_fn = (
        PACKAGE_DIR
        / '..'
        / f'legacy/nt_orig/DATAFILES/data36/maps/shoremap_{hemisphere}_25'
    )
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

    minic_path = PACKAGE_DIR / '..' / 'legacy/nt_orig/DATAFILES/data36/maps' / minic_fn
    minic = np.fromfile(minic_path, dtype='>i2')[150:].reshape(
        get_ps25_grid_shape(hemisphere=hemisphere)
    )

    return minic


def original_example(*, hemisphere: Hemisphere) -> xr.Dataset:
    """Return the concentration field example for f17_20180101."""
    date = dt.date(2018, 1, 1)
    raw_fns = {
        'h19': f'tb_f17_{date:%Y%m%d}_v4_{hemisphere[0].lower()}19h.bin',
        'v19': f'tb_f17_{date:%Y%m%d}_v4_{hemisphere[0].lower()}19v.bin',
        'v22': f'tb_f17_{date:%Y%m%d}_v4_{hemisphere[0].lower()}22v.bin',
        'h37': f'tb_f17_{date:%Y%m%d}_v4_{hemisphere[0].lower()}37h.bin',
        'v37': f'tb_f17_{date:%Y%m%d}_v4_{hemisphere[0].lower()}37v.bin',
    }

    tbs = {}
    grid_shape = get_ps25_grid_shape(hemisphere=hemisphere)
    for tb in raw_fns.keys():
        tbfn = raw_fns[tb]
        tbs[tb] = np.fromfile(
            Path('/share/apps/amsr2-cdr/cdr_testdata/nt_goddard_input_tbs/') / tbfn,
            dtype=np.int16,
        ).reshape(grid_shape)

    conc_ds = nasateam(
        tbs=tbs,
        sat='17_final',
        hemisphere=hemisphere,
        shoremap=_get_shoremap(hemisphere=hemisphere),
        minic=_get_minic(hemisphere=hemisphere),
        date=date,
    )

    return conc_ds


def amsr2_nasateam(*, date: dt.date, hemisphere: Hemisphere):
    """Compute sea ice concentration from AU_SI25 TBs."""
    xr_tbs = get_au_si_tbs(
        date=date,
        hemisphere=hemisphere,
        resolution='25',
    )

    tbs = {
        'h19': xr_tbs['h18'].data,
        'v19': xr_tbs['v18'].data,
        'v22': xr_tbs['v23'].data,
        'h37': xr_tbs['h36'].data,
        'v37': xr_tbs['v36'].data,
    }

    conc_ds = nasateam(
        tbs=tbs,
        sat='u2',
        hemisphere=hemisphere,
        shoremap=_get_shoremap(hemisphere=hemisphere),
        minic=_get_minic(hemisphere=hemisphere),
        date=date,
    )

    return conc_ds
