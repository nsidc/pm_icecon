import datetime as dt
from typing import Iterator

import pandas as pd

from pm_icecon._types import Hemisphere, ValidSatellites


def get_ps25_grid_shape(*, hemisphere: Hemisphere) -> tuple[int, int]:
    """Get the polar stereo 25km resolution grid size."""
    shape = {
        'north': (448, 304),
        'south': (332, 316),
    }[hemisphere]

    return shape


def get_ps12_grid_shape(*, hemisphere: Hemisphere) -> tuple[int, int]:
    """Get the polar stereo 12.5km resolution grid size."""
    shape = {
        'north': (896, 608),
        'south': (664, 632),
    }[hemisphere]

    return shape


# TODO: get rid of the other two ps* grid shape getters.
def get_ps_grid_shape(*, hemisphere: Hemisphere, resolution: str) -> tuple[int, int]:
    if resolution == '25':
        return get_ps25_grid_shape(hemisphere=hemisphere)
    elif resolution == '12':
        return get_ps12_grid_shape(hemisphere=hemisphere)
    else:
        raise NotImplementedError(f'No shape defined for {resolution=}')


def standard_output_filename(
    *,
    hemisphere: Hemisphere,
    date: dt.date,
    sat: ValidSatellites,
    resolution: str,
    algorithm: str,
    extension: str = '.nc',
) -> str:
    """Return a string representing the standard filename for bootstrap."""
    assert (
        extension[0] == '.'
    ), f'extension must contain `.`. Did you mean ".{extension}"?'
    return (
        f'{algorithm}_{hemisphere[0].upper()}H'
        f'_{date:%Y%m%d}_{sat}_{resolution}{extension}'
    )


def date_range(*, start_date: dt.date, end_date: dt.date) -> Iterator[dt.date]:
    """Yield a dt.date object representing each day between start_date and end_date."""
    for pd_timestamp in pd.date_range(start=start_date, end=end_date, freq='D'):
        yield pd_timestamp.date()
