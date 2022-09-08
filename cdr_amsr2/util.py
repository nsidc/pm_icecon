from cdr_amsr2._types import Hemisphere


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
