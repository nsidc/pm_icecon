import datetime as dt
from pathlib import Path

import numpy as np
import xarray as xr

from cdr_amsr2._types import Hemisphere
from cdr_amsr2.bt.masks import get_ps_invalid_ice_mask
from cdr_amsr2.constants import PACKAGE_DIR
from cdr_amsr2.fetch.au_si import AU_SI_RESOLUTIONS, get_au_si_tbs
from cdr_amsr2.nt.compute_nt_ic import nasateam
from cdr_amsr2.nt.masks import get_ps25_sst_mask
from cdr_amsr2.util import get_ps25_grid_shape, get_ps_grid_shape


def original_example(*, hemisphere: Hemisphere) -> xr.Dataset:
    """Return the concentration field example for f17_20180101."""

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

        minic_path = (
            PACKAGE_DIR / '..' / 'legacy/nt_orig/DATAFILES/data36/maps' / minic_fn
        )
        minic = np.fromfile(minic_path, dtype='>i2')[150:].reshape(
            get_ps25_grid_shape(hemisphere=hemisphere)
        )

        # Scale down by 10. The original alg. dealt w/ concentrations scaled by 10.
        minic = minic / 10

        return minic

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

    invalid_ice_mask = get_ps25_sst_mask(hemisphere=hemisphere, date=date)

    conc_ds = nasateam(
        tbs=tbs,
        sat='17_final',
        hemisphere=hemisphere,
        shoremap=_get_shoremap(hemisphere=hemisphere),
        minic=_get_minic(hemisphere=hemisphere),
        date=date,
        invalid_ice_mask=invalid_ice_mask,
    )

    return conc_ds


def amsr2_nasateam(
    *, date: dt.date, hemisphere: Hemisphere, resolution: AU_SI_RESOLUTIONS
):
    """Compute sea ice concentration from AU_SI25 TBs."""
    xr_tbs = get_au_si_tbs(
        date=date,
        hemisphere=hemisphere,
        resolution=resolution,
    )

    tbs = {
        'h19': xr_tbs['h18'].data,
        'v19': xr_tbs['v18'].data,
        'v22': xr_tbs['v23'].data,
        'h37': xr_tbs['h36'].data,
        'v37': xr_tbs['v36'].data,
    }

    shoremap = np.fromfile(
        (
            '/share/apps/amsr2-cdr/nasateam_ancillary/'
            f'shoremap_amsru_{hemisphere[0]}h{resolution}.dat'
        ),
        dtype=np.uint8,
    ).reshape(get_ps_grid_shape(hemisphere=hemisphere, resolution=resolution))
    minic = np.fromfile(
        (
            '/share/apps/amsr2-cdr/nasateam_ancillary/'
            f'minic_amsru_{hemisphere[0]}h{resolution}.dat'
        ),
        dtype=np.int16,
    ).reshape(get_ps_grid_shape(hemisphere=hemisphere, resolution=resolution))

    # Scale down by 10. The original alg. dealt w/ concentrations scaled by 10.
    minic = minic / 10

    # TODO: this function is currently defined in the bootstrap-specific masks
    # module. Should it be moved to the top-level masks? Originally split masks
    # between nt and bt modules because the original goddard nasateam example
    # used a unique invalid ice mask. Eventually won't matter too much because
    # we plan to move most masks into common nc files that will be read on a
    # per-grid basis.
    invalid_ice_mask = get_ps_invalid_ice_mask(
        hemisphere=hemisphere,
        date=date,
        resolution=resolution,
    )

    conc_ds = nasateam(
        tbs=tbs,
        sat='u2',
        hemisphere=hemisphere,
        shoremap=shoremap,
        minic=minic,
        date=date,
        invalid_ice_mask=invalid_ice_mask,
    )

    return conc_ds
