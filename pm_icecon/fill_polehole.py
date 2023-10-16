"""Routine(s) to fill the Northern Hemisphere pole hole.

fill_polehole.py

In general, these will be grid and sensor dependent
"""

import numpy as np


# TODO: This pole hole logic should be refactored.
#       Specifically, the definition of the pixels for which missing data
#       will be considered "pole hole" rather than simply "missing (because
#       of lack of sensor observation)" is on the same level of abstraction
#       as a "land_mask", and therefore should be identified and stored as
#       ancillary data in a similar location and with similar level of
#       description, including the derivation of the set of grid cells
#       identified as "pole hole".
def fill_pole_hole(conc):
    """Fill pole hole for Polar Stereo 12.5km grid.

    # Identify pole hole pixels for psn12.5
    # These pixels were identified by examining AUSI12-derived NH fields in 2021
    #    and are one ortho and diag from the commonly no-data pixels near
    #    the pole that year from AU_SI12 products
    """
    is_psn125 = conc.shape == (896, 608)

    if not is_psn125:
        raise ValueError(f'Could not determine pole hole for grid shape: {conc.shape}')

    pole_pixels = np.zeros((896, 608), dtype=np.uint8)
    pole_pixels[461, 304 : 311 + 1] = 1
    pole_pixels[462, 303 : 312 + 1] = 1
    pole_pixels[463, 302 : 313 + 1] = 1
    pole_pixels[464 : 471 + 1, 301 : 314 + 1] = 1
    pole_pixels[472, 302 : 313 + 1] = 1
    pole_pixels[473, 303 : 312 + 1] = 1
    pole_pixels[474, 304 : 311 + 1] = 1

    # Fill zeros or NaNs near the pole
    is_vals_near_pole = (pole_pixels == 1) & (conc > 0)
    is_missing_near_pole = (pole_pixels == 1) & ((conc == 0) | np.isnan(conc))
    mean_near_pole = np.mean(conc[is_vals_near_pole])
    conc[is_missing_near_pole] = mean_near_pole

    return conc
