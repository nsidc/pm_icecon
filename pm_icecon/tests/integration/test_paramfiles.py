import datetime as dt

import pm_icecon.bt.params.ausi_amsr2 as amsr2_params
from pm_icecon.bt.fields import get_bootstrap_fields
from pm_icecon.bt.params.experimental.ausi12_amsr2 import (
    get_ausi12_experimental_bootstrap_params,
)
from pm_icecon.bt.params.util import convert_to_pmicecon_bt_params
from pm_icecon.tests.unit.test_bt_params import _get_config_hash


def test_get_ausi12_experimental_bootstrap_params():
    date = dt.date(2020, 1, 1)
    satellite = "amsr2"
    gridid = "e2n25"

    bt_parameters = get_ausi12_experimental_bootstrap_params(
        date=date,
        satellite=satellite,
        gridid=gridid,
    )

    assert "wintrc" in bt_parameters.keys()


def test_get_bootstrap_fields():
    date = dt.date(2020, 1, 1)
    satellite = "amsr2"
    gridid = "e2n25"

    bt_field_dict = get_bootstrap_fields(
        date=date,
        satellite=satellite,
        gridid=gridid,
    )
    assert "invalid_ice_mask" in bt_field_dict.keys()


def test_convert_to_pmicecon_bt_params():
    hemisphere = "north"
    date = dt.date(2020, 1, 1)
    satellite = "amsr2"
    gridid = "e2n25"

    bt_parameters = get_ausi12_experimental_bootstrap_params(
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


def test_ausi12_amsr2_bt_params_north():
    date = dt.date(2022, 1, 1)
    fields = get_bootstrap_fields(date=date, satellite="amsr2", gridid="e2ns25")
    params = amsr2_params.get_ausi_bootstrap_params(
        date=date, satellite="amsr2", gridid="e2ns25"
    )
    bt_params = convert_to_pmicecon_bt_params(
        hemisphere="north", params=params, fields=fields
    )

    assert _get_config_hash(bt_params) == "8e61f93a2f762e962323f159342d282c"


def test_ausi12_amsr2_bt_params_south():
    date = dt.date(2022, 1, 1)
    fields = get_bootstrap_fields(date=date, satellite="amsr2", gridid="e2ss25")
    params = amsr2_params.get_ausi_bootstrap_params(
        date=date, satellite="amsr2", gridid="e2ss25"
    )
    bt_params = convert_to_pmicecon_bt_params(
        hemisphere="south", params=params, fields=fields
    )

    assert _get_config_hash(bt_params) == "a5e1d2959f31ad165a075cd9003fbdff"
