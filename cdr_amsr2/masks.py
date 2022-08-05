"""Code related to land, valid ice, and pole-hole masks."""
import datetime as dt
from pathlib import Path

import numpy as np
import numpy.typing as npt

from cdr_amsr2._types import Hemisphere
from cdr_amsr2.constants import PACKAGE_DIR


# TODO: accept `Hemisphere` arg and return None if South?
def get_ps_pole_hole_mask(*, resolution: str) -> npt.NDArray[np.bool_]:
    # values of 1 indicate the pole hole.
    if resolution == '25':
        pole_mask_psn = (
            np.fromfile(
                (
                    PACKAGE_DIR
                    / '../legacy/SB2_NRT_programs'
                    / '../SB2_NRT_programs/ANCILLARY/np_holemask.ssmi_f17'
                ).resolve(),
                dtype=np.int16,
            ).reshape(448, 304)
            == 1
        )
    elif resolution == '12':
        pole_mask_psn = (
            np.fromfile(
                Path(
                    '/share/apps/amsr2-cdr/cdr_testdata/btequiv_psn12.5/'
                    'bt_poleequiv_psn12.5km.dat'
                ),
                dtype=np.int16,
            ).reshape(896, 608)
            == 1
        )
    else:
        raise NotImplementedError(f'No pole hole mask for PS {resolution} available.')

    return pole_mask_psn


def get_ps_land_mask(
    *, hemisphere: Hemisphere, resolution: str
) -> npt.NDArray[np.bool_]:
    """Get the polar stereo 25km land mask."""
    # Ocean has a value of 0, land a value of 1, and coast a value of 2.
    if resolution == '25':
        shape = {
            'north': (448, 304),
            'south': (332, 316),
        }[hemisphere]
        _land_coast_array = np.fromfile(
            (
                PACKAGE_DIR
                / '../legacy/SB2_NRT_programs'
                / (
                    f'../SB2_NRT_programs/ANCILLARY/{hemisphere}_land_25'
                    # NOTE: According to scotts, the 'r' in the southern hemisphere
                    # filename probably stands for “revised“.
                    f"{'r' if hemisphere == 'south' else ''}"
                )
            ).resolve(),
            dtype=np.int16,
        ).reshape(shape)

        # TODO: land mask currently includes land and coast. Does this make sense? Are
        # we ever going to need to coast values? Maybe rename to `LAND_COAST_MASK`?
    elif resolution == '12':
        if hemisphere == 'south':
            raise NotImplementedError(
                f'No SH land masks for PS {resolution} available.'
            )

        _land_coast_array = np.fromfile(
            Path(
                '/share/apps/amsr2-cdr/cdr_testdata/btequiv_psn12.5/'
                'bt_landequiv_psn12.5km.dat'
            ),
            dtype=np.int16,
        ).reshape(896, 608)

    land_mask = _land_coast_array != 0

    return land_mask


def get_e2n625_land_mask() -> npt.NDArray[np.bool_]:
    """Get the northern hemisphere e2n625 land mask.

    The authoritative mask for the NH EASE2 Arctic subset
    is a 1680x1680 (centered) subset of the full 2880x2880 EASE2 NH grid
    These data were derived from the MCD12Q1 land type data set
    Encoding is
       50: Ocean (= water orthogonally connected to the global ocean)
      100: Lake (= water that is not connected to the global ocean)
      125: Coast (= "Land" that is adjacent to "Ocean"
      150: Land ("land type 1" that is not water and not ice)
      200: Ice (= land classification type "permanent ice"
    For the purposes of the Bootstrap algorithm:
       50 --> 0  (= "ocean" where sea ice might occur)
      all others ->
    """
    _land_coast_array_e2n625 = np.fromfile(
        Path('/share/apps/amsr2-cdr/bootstrap_masks/locli_e2n6.25_1680x1680.dat'),
        dtype=np.uint8,
    ).reshape(1680, 1680)

    # TODO: land mask currently includes land and coast. Does this make sense? Are
    # we ever going to need to coast values? Maybe rename to `LAND_COAST_MASK`?
    land_mask_e2n625 = _land_coast_array_e2n625 != 50

    return land_mask_e2n625


# TODO: rename to indicate this is derived from SST?
def get_ps25_valid_ice_mask(
    *,
    hemisphere: Hemisphere,
    date: dt.date,
) -> npt.NDArray[np.bool_]:
    """Read and return the polar stereo 25km valid ice mask."""
    if hemisphere == 'north':
        print('Reading valid ice mask for PSN 25km grid')
        sst_fn = (
            PACKAGE_DIR
            / '../legacy'
            / f'SB2_NRT_programs/ANCILLARY/np_sect_sst1_sst2_mask_{date:%m}.int'
        ).resolve()
        sst_mask = np.fromfile(sst_fn, dtype=np.int16).reshape(448, 304)
    else:
        print('Reading valid ice mask for PSN 25km grid')
        sst_fn = Path(
            '/share/apps/amsr2-cdr'
            '/cdr_testdata/bt_goddard_ANCILLARY'
            f'/SH_{date:%m}_SST_avhrr_threshold_{date:%m}_fixd.int'
        )
        sst_mask = np.fromfile(sst_fn, dtype=np.int16).reshape(332, 316)

    is_high_sst = sst_mask == 24

    return is_high_sst


def get_ps12_valid_ice_mask(
    hemisphere: Hemisphere,
    date: dt.date,
) -> npt.NDArray[np.bool_]:
    """Read and return the polar stereo 12km valid ice mask.

    scotts said on 2022-07-11:

    I've created NH 12.5km "bt_equiv" files for use with a version of the
    bootstrap algorithm running in the Northern Hemisphere on the 12.5km polar
    stereo grid.I placed these in:

    * /share/apps/amsr2-cdr/cdr_testdata/btequiv_psn12.5/

    * And if you do a 'tree' command on the new directory, tree btequiv_psn12.5/
      from the cdr_testdate/ dir, you get:

    btequiv_psn12.5/
    ├── bt_landequiv_psn12.5km.dat     # a landmask with the same
                                       #  encoding as the BT mask
    ├── bt_poleequiv_psn12.5km.dat     # a pole-hole mask with the same encoding
                                       #  as the BT polehole mask
    ├── bt_validmask_psn12.5km_01.dat  # 12 monthly valid ice masks
    ├── bt_validmask_psn12.5km_02.dat
    ├── bt_validmask_psn12.5km_03.dat
    ├── bt_validmask_psn12.5km_04.dat
    ├── bt_validmask_psn12.5km_05.dat
    ├── bt_validmask_psn12.5km_06.dat
    ├── bt_validmask_psn12.5km_07.dat
    ├── bt_validmask_psn12.5km_08.dat
    ├── bt_validmask_psn12.5km_09.dat
    ├── bt_validmask_psn12.5km_10.dat
    ├── bt_validmask_psn12.5km_11.dat
    └── bt_validmask_psn12.5km_12.dat
    0 directories, 14 files

    All of these are raw binary files that are 608x896 2-byte (16-bit) integers,
    i.e. numpy dtype "np.int16"
    """
    if hemisphere == 'south':
        raise NotImplementedError('We have not yet created a ps12 SH valid ice mask.')

    mask_fn = Path(
        '/share/apps/amsr2-cdr/cdr_testdata/btequiv_psn12.5/'
        f'bt_validmask_psn12.5km_{date:%m}.dat'
    )

    # The valid ice array has values of 0, 4, 24, and 32. It looks like values
    # of 4 represent areas of valid ice.
    valid_ice_arr = np.fromfile(mask_fn, dtype=np.int16).reshape(896, 608)

    # TODO: this is the same as the 25km mask. DRY it out!
    return valid_ice_arr == 24


def get_ps_valid_ice_mask(
    hemisphere: Hemisphere,
    date: dt.date,
    resolution: str,
) -> npt.NDArray[np.bool_]:
    if resolution == '12':
        return get_ps12_valid_ice_mask(hemisphere=hemisphere, date=date)
    elif resolution == '25':
        return get_ps25_valid_ice_mask(hemisphere=hemisphere, date=date)
    else:
        raise NotImplementedError(f'No valid ice masks for PS {resolution} available.')
