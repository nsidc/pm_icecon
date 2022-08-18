import datetime as dt

from cdr_amsr2.bt.compute_bt_ic import _get_wx_params
from cdr_amsr2.bt.params.seasonal_params import AMSR2_NORTH_PARAMS
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
