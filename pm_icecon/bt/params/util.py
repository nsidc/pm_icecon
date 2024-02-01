import datetime as dt


from pm_icecon.bt.compute_bt_ic import _get_wx_params as interpolate_bt_wx_params


def setup_bootstrap_params_dict(
    *,
    initial_params_dict: dict,
    date: dt.date,
) -> dict:
    """Create bootstrap params dict with defaults."""
    bt_params = initial_params_dict.copy()

    # Set standard bootstrap values
    # TODO: these should not be defined as defaults of the data model that
    # represents the resulting data structure, not hard-coded in this function.
    bt_params["add1"] = 0.0
    bt_params["add2"] = -2.0
    bt_params["minic"] = 10.0
    bt_params["maxic"] = 1.0
    bt_params["mintb"] = 10.0
    bt_params["maxtb"] = 320.0

    # Some definitions include seasonal values for wintrc, wslope, wxlimt
    if "wintrc" not in bt_params.keys():
        # weather_filter_seasons = bt_params['weather_filter_seasons']
        wfs = bt_params["weather_filter_seasons"]
        bt_weather_params_struct = interpolate_bt_wx_params(
            date=date,
            weather_filter_seasons=wfs,
        )
        bt_params["wintrc"] = bt_weather_params_struct.wintrc
        bt_params["wslope"] = bt_weather_params_struct.wslope
        bt_params["wxlimt"] = bt_weather_params_struct.wxlimt

    return bt_params
