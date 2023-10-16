import datetime as dt

from pm_icecon.bt.masks import get_ps_invalid_ice_mask
from pm_icecon.gridid import get_gridid_hemisphere, get_gridid_resolution
from pm_icecon.masks import get_ps_land_mask, get_ps_pole_hole_mask


def get_bootstrap_fields(
    *,
    date: dt.date,
    satellite: str,
    # TODO: replace `gridid` with hemisphere & resolution kwargs. Only the
    # top-most API of the ECDR should deal with translating gridid into
    # appropriate values for downstream functions.
    gridid: str,
):
    hemisphere = get_gridid_hemisphere(gridid)
    resolution = get_gridid_resolution(gridid)

    invalid_ice_mask = get_ps_invalid_ice_mask(
        hemisphere=hemisphere,
        date=date,
        resolution=resolution,  # type: ignore[arg-type]
    )

    land_mask = get_ps_land_mask(hemisphere=hemisphere, resolution=resolution)

    # There's no pole hole in the southern hemisphere.
    pole_mask = (
        get_ps_pole_hole_mask(resolution=resolution) if hemisphere == 'north' else None
    )

    return dict(
        invalid_ice_mask=invalid_ice_mask,
        land_mask=land_mask,
        pole_mask=pole_mask,
    )
