import datetime as dt

import pm_icecon.bt.params.ausi_amsr2 as amsr2_params
import pm_icecon.bt.params.ausi_amsre as amsre_params
from pm_icecon.bt.params.experimental.ausi12_amsr2 import (
    get_ausi12_experimental_bootstrap_params,
)
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


def test_ausi12_amsr2_bt_params_north():
    date = dt.date(2022, 1, 1)
    params = amsr2_params.get_ausi_amsr2_bootstrap_params(
        date=date, satellite="amsr2", gridid="e2ns25"
    )

    assert _get_config_hash(params) == "af19cc9a1d289c97b0b5cc9dd1bf952c"


def test_ausi12_amsr2_bt_params_south():
    date = dt.date(2022, 1, 1)
    params = amsr2_params.get_ausi_amsr2_bootstrap_params(
        date=date, satellite="amsr2", gridid="e2ss25"
    )

    assert _get_config_hash(params) == "74764a3d75261eea42a1ee703b5123c2"


def test_ausi12_amsre_bt_params_north():
    date = dt.date(2022, 1, 1)
    params = amsre_params.get_ausi_amsre_bootstrap_params(
        date=date, satellite="amsre", gridid="e2ns25"
    )

    assert _get_config_hash(params) == "af19cc9a1d289c97b0b5cc9dd1bf952c"


def test_ausi12_amsre_bt_params_south():
    date = dt.date(2022, 1, 1)
    params = amsre_params.get_ausi_amsre_bootstrap_params(
        date=date, satellite="amsre", gridid="e2ss25"
    )

    assert _get_config_hash(params) == "74764a3d75261eea42a1ee703b5123c2"
