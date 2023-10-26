import datetime as dt

from pm_tb_data._types import NORTH

from pm_icecon.util import standard_output_filename


def test_standard_output_filename():
    expected = "alg_NH_20210203_monthly_amsr2_25.nc"
    actual = standard_output_filename(
        hemisphere=NORTH,
        date=dt.date(2021, 2, 3),
        sat="amsr2",
        resolution="25",
        algorithm="alg",
        timeframe="monthly",
    )

    assert actual == expected
