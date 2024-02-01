"""Code for working with grid identifiers.

A grid identifier is shorthand for a particular grid and projection definition.
"""
from pm_tb_data._types import Hemisphere
from pm_tb_data.fetch.amsr.util import AMSR_RESOLUTIONS


# Note: these routines are also in seaice_ecdr's gridid_to_xr_dataarray.py
# TODO: de-duplicate this code!!! This code should belong with the ECDR, which
# has specific grids defined that map to these gridids. Other users of this code
# may have completely differently formatted data. It shouldn't matter what the
# grid is from this library's perspective!
def get_gridid_hemisphere(gridid: str) -> Hemisphere:
    # Return the hemisphere of the gridid
    if "psn" in gridid:
        return "north"
    elif "e2n" in gridid:
        return "north"
    elif "pss" in gridid:
        return "south"
    elif "e2s" in gridid:
        return "south"
    else:
        raise ValueError(f"Could not find hemisphere for gridid: {gridid}")


def get_gridid_resolution(gridid: str) -> AMSR_RESOLUTIONS:
    if "12.5" in gridid:
        return "12"
    elif "25" in gridid:
        return "25"
    else:
        raise ValueError(f"Could not find resolution for gridid: {gridid}")
