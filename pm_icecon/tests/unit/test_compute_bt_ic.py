import datetime as dt

from pm_icecon.bt.compute_bt_ic import _get_wx_params
from pm_icecon.bt.params.amsr2 import AMSR2_NORTH_PARAMS
from pm_icecon.bt.params.cetbv2_amsr2 import (
    A2L1C_NORTH_PARAMS,
    _ret_parameters_amsru2_f_params,
)
from pm_icecon.bt.params.class_sats import (
    OTHER_NORTH_PARAMS,
    SMMR_NORTH_PARAMS,
    SSMIS_NORTH_PARAMS,
)
from pm_icecon.config.models.bt import WeatherFilterParams


def test__get_wx_params_in_between_seasons():
    """Test that interpolation occurs between seasons."""
    # replicate season 2 from `ret_parameters_amsru2`
    date = dt.date(2020, 5, 23)
    expected = WeatherFilterParams(
        wintrc=((date.day / 32.0) * (82.71 - 84.73)) + 84.73,
        wslope=0.5352,
        # These are the values for the original Goddard values
        # wxlimt=((date.day / 32.0) * (23.34 - 18.39)) + 18.39,
        # These are the values for the CDR-derived values
        wxlimt=((date.day / 32.0) * (21.7 - 13.7)) + 13.7,
    )
    actual = _get_wx_params(
        date=date,
        weather_filter_seasons=(
            AMSR2_NORTH_PARAMS['weather_filter_seasons']  # type: ignore[arg-type]
        ),
    )

    try:
        assert expected == actual
    except AssertionError as e:
        print('Failed expected == actual assertion')
        print(f'expected:\n{expected}')
        print(f'actual:\n{actual}')
        raise e


def test__get_wx_params_wrap_around_seasons():
    """Test that interpolation occurs between seasons that cross December."""
    # replicate season 4 from `ret_parameters_amsru2`
    date = dt.date(2020, 10, 12)
    expected = WeatherFilterParams(
        wintrc=((date.day / 32.0) * (84.73 - 82.71)) + 82.71,
        wslope=0.5352,
        # Original Goddard values:
        # wxlimt=((date.day / 32.0) * (18.39 - 23.34)) + 23.34,
        # CDR-calculated values:
        wxlimt=((date.day / 32.0) * (13.7 - 21.7)) + 21.7,
    )
    actual = _get_wx_params(
        date=date,
        weather_filter_seasons=(
            AMSR2_NORTH_PARAMS['weather_filter_seasons']  # type: ignore[arg-type]
        ),
    )

    assert expected == actual


def test__get_wx_params_for_provided_season():
    date = dt.date(2020, 1, 16)
    # Original Goddard values:
    expected = WeatherFilterParams(
        wintrc=84.73,
        wslope=0.5352,
        wxlimt=18.39,
    )
    # CDR-calculated values:
    expected = WeatherFilterParams(
        wintrc=84.73,
        wslope=0.5352,
        wxlimt=13.7,
    )
    actual = _get_wx_params(
        date=date,
        weather_filter_seasons=(
            AMSR2_NORTH_PARAMS['weather_filter_seasons']  # type: ignore[arg-type]
        ),
    )

    assert expected == actual


def test__get_wx_params_for_smmr():
    date = dt.date(2020, 6, 14)
    expected = WeatherFilterParams(
        wintrc=60.1667,
        wslope=0.633333,
        wxlimt=24.00,
    )
    actual = _get_wx_params(
        date=date,
        weather_filter_seasons=(
            SMMR_NORTH_PARAMS['weather_filter_seasons']  # type: ignore[arg-type]
        ),
    )

    assert expected == actual


def test__get_wx_params_for_amsru_from_goddard():
    date = dt.date(2020, 5, 14)
    expected = WeatherFilterParams(
        wintrc=84.73,
        wslope=0.5352,
        wxlimt=18.39,
    )
    wx_filter_seasons = _ret_parameters_amsru2_f_params['weather_filter_seasons']
    actual = _get_wx_params(
        date=date,
        weather_filter_seasons=(wx_filter_seasons),  # type: ignore[arg-type]
    )

    assert expected == actual


def test__get_wx_params_for_a2l1c():
    date = dt.date(2020, 5, 14)
    expected = WeatherFilterParams(
        wintrc=84.73,
        wslope=0.5352,
        wxlimt=13.7,
    )
    actual = _get_wx_params(
        date=date,
        weather_filter_seasons=(
            A2L1C_NORTH_PARAMS['weather_filter_seasons']  # type: ignore[arg-type]
        ),
    )

    assert expected == actual


def test__get_wx_params_for_f17f18():
    date = dt.date(2020, 10, 15)
    expected = WeatherFilterParams(
        wintrc=89.2000,
        wslope=0.503750,
        wxlimt=21.0,
    )
    actual = _get_wx_params(
        date=date,
        weather_filter_seasons=(
            SSMIS_NORTH_PARAMS['weather_filter_seasons']  # type: ignore[arg-type]
        ),
    )

    assert expected == actual


def test__get_wx_params_for_other():
    date = dt.date(2020, 10, 16)
    expected = WeatherFilterParams(
        wintrc=90.3355,
        wslope=0.501537,
        wxlimt=14.0,
    )
    actual = _get_wx_params(
        date=date,
        weather_filter_seasons=(
            OTHER_NORTH_PARAMS['weather_filter_seasons']  # type: ignore[arg-type]
        ),
    )

    assert expected == actual
