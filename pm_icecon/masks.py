"""Code related to land, valid ice, and pole-hole masks.

Algorithm-specific valid ice masks are in their respective algorithm-specific
modules (e.g., `bt.masks`).

TODO: should we really use different valid ice masks depending on the algorithm?
Why not one source of truth? Ultimatley we do want to make valid ice masks
configurable for the user. Perhaps these should me moved into the `params`
subpackage (e.g., `bt.params`)
"""
import datetime as dt

import numpy as np
import numpy.typing as npt
from pm_tb_data.fetch.au_si import AU_SI_RESOLUTIONS

from pm_icecon._types import Hemisphere
from pm_icecon.constants import (
    BOOTSTRAP_MASKS_DIR,
    BT_GODDARD_ANCILLARY_DIR,
    CDR_TESTDATA_DIR,
)
from pm_icecon.util import get_ps25_grid_shape


# TODO: accept `Hemisphere` arg and return None if South?
def get_ps_pole_hole_mask(*, resolution: AU_SI_RESOLUTIONS) -> npt.NDArray[np.bool_]:
    # values of 1 indicate the pole hole.
    if resolution == "25":
        pole_mask_psn = (
            np.fromfile(
                (BT_GODDARD_ANCILLARY_DIR / "np_holemask.ssmi_f17").resolve(),
                dtype=np.int16,
            ).reshape(448, 304)
            == 1
        )
    elif resolution == "12":
        pole_mask_psn = (
            np.fromfile(
                CDR_TESTDATA_DIR / "btequiv_psn12.5/bt_poleequiv_psn12.5km.dat",
                dtype=np.int16,
            ).reshape(896, 608)
            == 1
        )
    else:
        raise NotImplementedError(f"No pole hole mask for PS {resolution} available.")

    return pole_mask_psn


def get_pss_12_validice_land_coast_array(*, date: dt.date) -> npt.NDArray[np.int16]:
    """Get the polar stereo south 12.5km valid ice/land/coast array.

    4 unique values:
        * 0 == land
        * 4 == valid ice
        * 24 == invalid ice
        * 32 == coast.
    """
    fn = BOOTSTRAP_MASKS_DIR / f"bt_valid_pss12.5_int16_{date:%m}.dat"
    validice_land_coast = np.fromfile(fn, dtype=np.int16).reshape(664, 632)

    return validice_land_coast


def get_ps_land_mask(
    *,
    hemisphere: Hemisphere,
    resolution: AU_SI_RESOLUTIONS,
) -> npt.NDArray[np.bool_]:
    """Get the polar stereo 25km land mask."""
    # Ocean has a value of 0, land a value of 1, and coast a value of 2.
    if resolution == "25":
        shape = get_ps25_grid_shape(hemisphere=hemisphere)
        _land_coast_array = np.fromfile(
            (
                BT_GODDARD_ANCILLARY_DIR
                / (
                    f"{hemisphere}_land_25"
                    # NOTE: According to scotts, the 'r' in the southern hemisphere
                    # filename probably stands for “revised“.
                    f"{'r' if hemisphere == 'south' else ''}"
                )
            ).resolve(),
            dtype=np.int16,
        ).reshape(shape)

        land_mask = _land_coast_array != 0

        # TODO: land mask currently includes land and coast. Does this make sense? Are
        # we ever going to need to coast values? Maybe rename to `LAND_COAST_MASK`?
    elif resolution == "12":
        if hemisphere == "south":
            # Any date is OK. The land mask is the same for all of the pss 12
            # validice/land masks
            _land_coast_array = get_pss_12_validice_land_coast_array(
                date=dt.date.today()
            )
            land_mask = np.logical_or(_land_coast_array == 0, _land_coast_array == 32)
        else:
            _land_coast_array = np.fromfile(
                CDR_TESTDATA_DIR / "btequiv_psn12.5/bt_landequiv_psn12.5km.dat",
                dtype=np.int16,
            ).reshape(896, 608)

            land_mask = _land_coast_array != 0

    return land_mask


def get_e2n625_land_mask(anc_dir) -> npt.NDArray[np.bool_]:
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
        anc_dir / "locli_e2n6.25_1680x1680.dat",
        dtype=np.uint8,
    ).reshape(1680, 1680)

    # TODO: land mask currently includes land and coast. Does this make sense? Are
    # we ever going to need to coast values? Maybe rename to `LAND_COAST_MASK`?
    land_mask_e2n625 = _land_coast_array_e2n625 != 50

    return land_mask_e2n625
