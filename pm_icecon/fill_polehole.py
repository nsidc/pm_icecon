"""
fill_polehole.py

Routine(s) to fill the Northern Hemisphere pole hole

In general, these will be grid and sensor dependent
"""

import numpy as np


def fill_pole_hole(conc):
    is_psn125 = conc.shape == (896, 608)

    if not is_psn125:
        raise ValueError(f'Could not determine pole hole for grid shape: {conc.shape}')

    # Identify pole hole pixels for psn12.5
    # These pixels were identified by examining AUSI12-derived NH fields in 2021
    #    and are one ortho and diag from the commonly no-data pixels near
    #    the pole that year from AU_SI12 products
    pole_pixels = np.zeros((896, 608), dtype=np.uint8)
    pole_pixels[461, 304:311+1] = 1
    pole_pixels[462, 303:312+1] = 1
    pole_pixels[463, 302:313+1] = 1
    pole_pixels[464:471+1, 301:314+1] = 1
    pole_pixels[472, 302:313+1] = 1
    pole_pixels[473, 303:312+1] = 1
    pole_pixels[474, 304:311+1] = 1

    is_vals_near_pole = (pole_pixels == 1) & (conc > 0)
    is_missing_near_pole = (pole_pixels == 1) & (conc == 0)
    mean_near_pole = np.mean(conc[is_vals_near_pole])
    conc[is_missing_near_pole] = mean_near_pole

    return conc
