import datetime as dt
from pathlib import Path

import numpy as np

import cdr_amsr2.nt.compute_nt_ic as nt
from cdr_amsr2.fetch.au_si25 import get_au_si25_tbs

if __name__ == '__main__':
    # TODO: this obviously isn't right...still need to get the new tiepoints for
    # AMSR2
    sat = 'f17'
    # TODO: use `Hemisphere` type.
    hem = 'n'

    xr_tbs = get_au_si25_tbs(
        base_dir=Path('/ecs/DP1/AMSA/AU_SI25.001/'),
        date=dt.date(2022, 5, 4),
        hemisphere='north',
    )

    # TODO: we'll eventually want to make the following function calls from the `nt`
    # module generic enough to accept tb arrays without fixed channel names (e.g,
    # `tb1` and `tb2` vs `v19` and `h19`. Also, support xarray datasets/data arrays!

    tbs = {
        'v19': xr_tbs['v18'].data,
        'h19': xr_tbs['h18'].data,
        'v22': xr_tbs['v23'].data,
        'h22': xr_tbs['h23'].data,
        'v37': xr_tbs['v36'].data,
        'h37': xr_tbs['h36'].data,
    }

    # Spatial interpolation.
    # TODO: can this be extracted to a common module that might be used by both nt
    # and bt algorithms in the future?
    tbs = nt.nt_spatint(tbs)

    # TODO: this obviously isn't right...still need to get the new tiepoints for
    # AMSR2
    tiepoints = nt.get_tiepoints(sat, hem)
    nt_coefficients = nt.compute_nt_coefficients(tiepoints)

    is_valid_tbs = nt.compute_valid_tbs(tbs)
    gr_thresholds = nt.get_gr_thresholds(sat, hem)

    ratios = nt.compute_ratios(tbs, nt_coefficients)

    weather_filtered = nt.compute_weather_filtered(tbs, ratios, gr_thresholds)

    conc = nt.compute_nt_conc(tbs, nt_coefficients, ratios)

    # Set invalid tbs and weather-filtered values
    conc[~is_valid_tbs] = -10
    conc[weather_filtered] = 0
    conc = conc.astype(np.int16)

    # Apply NT-land spillover filter
    conc = nt.apply_nt_spillover(conc)

    # Apply SST-threshold
    conc = nt.apply_sst(conc)

    # Apply pole hole
    conc = nt.apply_polehole(conc)
