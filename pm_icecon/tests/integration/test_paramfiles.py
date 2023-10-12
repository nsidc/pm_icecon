import datetime as dt

from pm_icecon.bt.fields import get_bootstrap_fields
from pm_icecon.bt.params.experimental.ausi12_amsr2 import get_bootstrap_params
from pm_icecon.bt.params.util import convert_to_pmicecon_bt_params


def test_get_bootstrap_params():
    date = dt.date(2020, 1, 1)
    satellite = 'amsr2'
    gridid = 'e2n25'

    bt_parameters = get_bootstrap_params(
        date=date,
        satellite=satellite,
        gridid=gridid,
    )

    assert 'wintrc' in bt_parameters.keys()


def test_get_bootstrap_fields():
    date = dt.date(2020, 1, 1)
    satellite = 'amsr2'
    gridid = 'e2n25'

    bt_field_dict = get_bootstrap_fields(
        date=date,
        satellite=satellite,
        gridid=gridid,
    )
    assert 'invalid_ice_mask' in bt_field_dict.keys()


def test_convert_to_pmicecon_bt_params():
    hemisphere = 'north'
    date = dt.date(2020, 1, 1)
    satellite = 'amsr2'
    gridid = 'e2n25'

    bt_parameters = get_bootstrap_params(
        date=date,
        satellite=satellite,
        gridid=gridid,
    )
    bt_field_dict = get_bootstrap_fields(
        date=date,
        satellite=satellite,
        gridid=gridid,
    )
    oldstyle_params = convert_to_pmicecon_bt_params(
        hemisphere, bt_parameters, bt_field_dict
    )
    assert oldstyle_params is not None
