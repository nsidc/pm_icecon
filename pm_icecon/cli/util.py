import datetime as dt


def datetime_to_date(_ctx, _param, value: dt.datetime) -> dt.date:
    """Click callback that takes a `dt.datetime` and returns `dt.date`."""
    return value.date()
