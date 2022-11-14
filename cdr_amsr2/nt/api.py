import datetime as dt

import numpy as np
import numpy.typing as npt

from cdr_amsr2._types import Hemisphere
from cdr_amsr2.bt.masks import get_ps_invalid_ice_mask
from cdr_amsr2.constants import CDR_TESTDATA_DIR
from cdr_amsr2.fetch.au_si import AU_SI_RESOLUTIONS, get_au_si_tbs
from cdr_amsr2.interpolation import spatial_interp_tbs
from cdr_amsr2.nt.compute_nt_ic import nasateam
from cdr_amsr2.util import get_ps_grid_shape


def amsr2_nasateam(
    *, date: dt.date, hemisphere: Hemisphere, resolution: AU_SI_RESOLUTIONS
):
    """Compute sea ice concentration from AU_SI25 TBs."""
    xr_tbs = get_au_si_tbs(
        date=date,
        hemisphere=hemisphere,
        resolution=resolution,
    )

    tbs: dict[str, npt.NDArray] = {
        'h19': xr_tbs['h18'].data,
        'v19': xr_tbs['v18'].data,
        'v22': xr_tbs['v23'].data,
        'h37': xr_tbs['h36'].data,
        'v37': xr_tbs['v36'].data,
    }

    # interpolate tbs
    for tb in tbs.keys():
        tbs[tb] = spatial_interp_tbs(tbs[tb])

    _nasateam_ancillary_dir = CDR_TESTDATA_DIR / 'nasateam_ancillary'
    shoremap = np.fromfile(
        (_nasateam_ancillary_dir / f'shoremap_amsru_{hemisphere[0]}h{resolution}.dat'),
        dtype=np.uint8,
    ).reshape(get_ps_grid_shape(hemisphere=hemisphere, resolution=resolution))
    minic = np.fromfile(
        (_nasateam_ancillary_dir / f'minic_amsru_{hemisphere[0]}h{resolution}.dat'),
        dtype=np.int16,
    ).reshape(get_ps_grid_shape(hemisphere=hemisphere, resolution=resolution))

    # Scale down by 10. The original alg. dealt w/ concentrations scaled by 10.
    minic = minic / 10  # type: ignore[assignment]

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
