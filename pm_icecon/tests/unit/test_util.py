import datetime as dt

from pm_tb_data._types import NORTH

from pm_icecon.util import date_range, standard_output_filename


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


def test_date_range():
    start_date = dt.date(2021, 1, 2)
    end_date = dt.date(2021, 1, 5)
    expected = [
        start_date,
        dt.date(2021, 1, 3),
        dt.date(2021, 1, 4),
        end_date,
    ]
    actual = list(date_range(start_date=start_date, end_date=end_date))

    assert expected == actual
