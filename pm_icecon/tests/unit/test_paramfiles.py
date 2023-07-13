import datetime as dt

from pm_icecon.bt.bt_params import (
    get_bootstrap_params,
    get_bootstrap_fields,
    convert_to_pmicecon_bt_params,
)


def test_get_bootstrap_params():
    date = dt.date(2020, 1, 1)
    satellite = 'amsr2'
    gridid = 'e2n25'

    bt_parameters = get_bootstrap_params(date, satellite, gridid)

    assert 'wintrc' in bt_parameters.keys()



def test_get_bootstrap_fields():
    date = dt.date(2020, 1, 1)
    satellite = 'amsr2'
    gridid = 'e2n25'

    bt_field_dict = get_bootstrap_fields(date, satellite, gridid)

    assert 'invalid_ice_mask' in bt_field_dict.keys()


def test_convert_to_pmicecon_bt_params():
    hemisphere = 'north'
    date = dt.date(2020, 1, 1)
    satellite = 'amsr2'
    gridid = 'e2n25'

    bt_parameters = get_bootstrap_params(date, satellite, gridid)
    bt_field_dict = get_bootstrap_fields(date, satellite, gridid)
    oldstyle_params = convert_to_pmicecon_bt_params(
        hemisphere, bt_parameters, bt_fields)

    assert oldstyle_params is not None
