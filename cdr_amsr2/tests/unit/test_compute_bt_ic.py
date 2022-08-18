import datetime as dt

from cdr_amsr2.bt.compute_bt_ic import _get_wx_params
from cdr_amsr2.bt.params.seasonal_params import AMSR2_NORTH_PARAMS


def test__get_wx_params():
    # replicate season 2 from `ret_parameters_amsru2`
    expected = 
    actual = _get_wx_params(
        date=dt.date(),
        weather_filter_seasons=AMSR2_NORTH_PARAMS.weather_filter_seasons,
    )
