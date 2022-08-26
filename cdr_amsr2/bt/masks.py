import datetime as dt
from pathlib import Path

import numpy as np
import numpy.typing as npt

from cdr_amsr2._types import Hemisphere
from cdr_amsr2.constants import PACKAGE_DIR
from cdr_amsr2.fetch.au_si import AU_SI_RESOLUTIONS
from cdr_amsr2.masks import get_pss_12_validice_land_coast_array
from cdr_amsr2.util import get_ps25_grid_shape


def get_ps_valid_ice_mask(
    *,
    hemisphere: Hemisphere,
    date: dt.date,
    resolution: AU_SI_RESOLUTIONS,
) -> npt.NDArray[np.bool_]:
    """Read and return the polar stereo valid ice mask."""
    print(f'Reading valid ice mask for PS{hemisphere[0].upper()} {resolution}km grid')
    if hemisphere == 'north':
        if resolution == '25':
            sst_fn = (
                PACKAGE_DIR
                / '../legacy'
                / f'SB2_NRT_programs/ANCILLARY/np_sect_sst1_sst2_mask_{date:%m}.int'
            ).resolve()
            sst_mask = np.fromfile(sst_fn, dtype=np.int16).reshape(
                get_ps25_grid_shape(hemisphere=hemisphere)
            )
        elif resolution == '12':
            mask_fn = Path(
                '/share/apps/amsr2-cdr/cdr_testdata/btequiv_psn12.5/'
                f'bt_validmask_psn12.5km_{date:%m}.dat'
            )

            sst_mask = np.fromfile(mask_fn, dtype=np.int16).reshape(896, 608)
    else:
        if resolution == '12':
            # values of 24 indicate invalid ice.
            sst_mask = get_pss_12_validice_land_coast_array(date=date)
        elif resolution == '25':
            sst_fn = Path(
                '/share/apps/amsr2-cdr'
                '/cdr_testdata/bt_goddard_ANCILLARY'
                f'/SH_{date:%m}_SST_avhrr_threshold_{date:%m}_fixd.int'
            )
            sst_mask = np.fromfile(sst_fn, dtype=np.int16).reshape(
                get_ps25_grid_shape(hemisphere=hemisphere)
            )

    is_high_sst = sst_mask == 24

    return is_high_sst
