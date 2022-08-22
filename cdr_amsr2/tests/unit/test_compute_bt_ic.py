import datetime as dt

from cdr_amsr2.bt.compute_bt_ic import _get_wx_params
from cdr_amsr2.bt.params.a2l1c import A2L1C_NORTH_PARAMS
from cdr_amsr2.bt.params.amsr2 import AMSR2_NORTH_PARAMS
from cdr_amsr2.bt.params.dmsp import F17_F18_NORTH_PARAMS
from cdr_amsr2.bt.params.others import OTHER_NORTH_PARAMS
from cdr_amsr2.bt.params.smmr import SMMR_NORTH_PARAMS
from cdr_amsr2.config.models.bt import WeatherFilterParams


def test__get_wx_params_in_between_seasons():
    """Test that interpolation occurs between seasons."""
    # replicate season 2 from `ret_parameters_amsru2`
    date = dt.date(2020, 5, 23)
    expected = WeatherFilterParams(
        wintrc=((date.day / 32.0) * (82.71 - 84.73)) + 84.73,
        wslope=0.5352,
        wxlimt=((date.day / 32.0) * (23.34 - 18.39)) + 18.39,
    )
    actual = _get_wx_params(
        date=date,
        weather_filter_seasons=AMSR2_NORTH_PARAMS.weather_filter_seasons,
    )

    assert expected == actual


def test__get_wx_params_wrap_around_seasons():
    """Test that interpolation occurs between seasons that cross December."""
    # replicate season 4 from `ret_parameters_amsru2`
    date = dt.date(2020, 10, 12)
    expected = WeatherFilterParams(
        wintrc=((date.day / 32.0) * (84.73 - 82.71)) + 82.71,
        wslope=0.5352,
        wxlimt=((date.day / 32.0) * (18.39 - 23.34)) + 23.34,
    )
    actual = _get_wx_params(
        date=date,
        weather_filter_seasons=AMSR2_NORTH_PARAMS.weather_filter_seasons,
    )

    assert expected == actual


def test__get_wx_params_for_provided_season():
    date = dt.date(2020, 1, 16)
    expected = WeatherFilterParams(
        wintrc=84.73,
        wslope=0.5352,
        wxlimt=18.39,
    )
    actual = _get_wx_params(
        date=date,
        weather_filter_seasons=AMSR2_NORTH_PARAMS.weather_filter_seasons,
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
        weather_filter_seasons=SMMR_NORTH_PARAMS.weather_filter_seasons,
    )

    assert expected == actual


def test__get_wx_params_for_a2l1c():
    date = dt.date(2020, 5, 14)
    expected = WeatherFilterParams(
        wintrc=84.73,
        wslope=0.5352,
        wxlimt=18.39,
    )
    actual = _get_wx_params(
        date=date,
        weather_filter_seasons=A2L1C_NORTH_PARAMS.weather_filter_seasons,
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
        weather_filter_seasons=F17_F18_NORTH_PARAMS.weather_filter_seasons,
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
        weather_filter_seasons=OTHER_NORTH_PARAMS.weather_filter_seasons,
    )

    assert expected == actual
