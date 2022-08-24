import datetime as dt

from cdr_amsr2._types import Hemisphere, ValidSatellites


# TODO: move to top-level module. This could take `algorithm` (nt or bt),
# `sensor` (amsr2, SSMI, etc).
def standard_output_filename(
    *,
    hemisphere: Hemisphere,
    date: dt.date,
    sat: ValidSatellites,
    resolution: str,
    extension: str = '.nc',
) -> str:
    """Return a string representing the standard filename for bootstrap."""
    assert (
        extension[0] == '.'
    ), f'extension must contain `.`. Did you mean ".{extension}"?'
    return f'bt_{hemisphere[0].upper()}H_{date:%Y%m%d}_{sat}_{resolution}{extension}'
