import datetime as dt

from pm_icecon.bt.bt_params import (
    convert_to_pmicecon_bt_params,
    get_bootstrap_fields,
    get_bootstrap_params,
)


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

    try:
        bt_field_dict = get_bootstrap_fields(
            date=date,
            satellite=satellite,
            gridid=gridid,
        )
        assert 'invalid_ice_mask' in bt_field_dict.keys()
    except FileNotFoundError:
        print('Skipping test_get_bootstrap_params() because file not found')


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
    try:
        bt_field_dict = get_bootstrap_fields(
            date=date,
            satellite=satellite,
            gridid=gridid,
        )
        oldstyle_params = convert_to_pmicecon_bt_params(
            hemisphere, bt_parameters, bt_field_dict
        )
        assert oldstyle_params is not None
    except FileNotFoundError:
        print('Skipping test_get_bootstrap_params() because file not found')
