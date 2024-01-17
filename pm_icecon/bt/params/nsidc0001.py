"""NSIDC0001 Bootstrap parameters.

Bootstrap parameters for use with TBs derived from NSIDC-0001

Parameters are based on values from Bootstrap code in cdralgos:
  https://bitbucket.org/nsidc/cdralgos/src/master/bt_cdr/ret_parameters_cdr.f
"""
import datetime as dt

from pm_tb_data._types import Hemisphere

from pm_icecon.bt._types import Line, Tiepoint
from pm_icecon.bt.params._types import ParamsDict
from pm_icecon.bt.params.util import setup_bootstrap_params_dict
from pm_icecon.config.models.bt import (
    BootstrapParams,
    TbSetParams,
    WeatherFilterParams,
    WeatherFilterParamsForSeason,
)
from pm_icecon.gridid import get_gridid_hemisphere

# GODDARD_AMSR2_NORTH_PARAMS = ParamsDict(
NSIDC0001_F17_NORTH_PARAMS = ParamsDict(
    vh37_params=TbSetParams(
        # water_tie_point_set=(Tiepoint(207.2), Tiepoint(131.9)),
        # ice_tie_point_set=(Tiepoint(256.3), Tiepoint(241.2)),
        # lnline=Line(offset=-71.99, slope=1.20),
        water_tie_point_set=(Tiepoint(201.916), Tiepoint(132.815)),
        ice_tie_point_set=(Tiepoint(255.670), Tiepoint(241.713)),
        lnline=Line(offset=-73.5471, slope=1.21104),
        # iceline=Line(offset=-25.9729, slope=1.04382),  # not yet used?
    ),
    v1937_params=TbSetParams(
        # water_tie_point_set=(Tiepoint(207.2), Tiepoint(182.4)),
        # ice_tie_point_set=(Tiepoint(256.3), Tiepoint(258.9)),
        # lnline=Line(offset=48.26, slope=0.8048),
        water_tie_point_set=(Tiepoint(201.916), Tiepoint(178.771)),
        ice_tie_point_set=(Tiepoint(255.670), Tiepoint(258.341)),
        lnline=Line(offset=47.0061, slope=0.809335),
        # iceline=Line(offset=112.803, slope=0.550296),  # not yet used?
    ),
    # NSIDC-0001 SSMI and SSMIS have transition months May and Oct
    weather_filter_seasons=[
        # November through April (`seas=1` in `ret_parameters_cdr.f`)
        WeatherFilterParamsForSeason(
            start_month=11,
            end_month=4,
            weather_filter_params=WeatherFilterParams(
                # wintrc=84.73,
                # wslope=0.5352,
                # wxlimt=18.39,
                wintrc=87.6467,
                wslope=0.517333,
                wxlimt=14.00,
            ),
        ),
        # May (`seas=2`) will get interpolated from the previous and next season
        # June through Sept. (`seas=3`)
        WeatherFilterParamsForSeason(
            start_month=6,
            end_month=9,
            weather_filter_params=WeatherFilterParams(
                # wintrc=82.71,
                # wslope=0.5352,
                # wxlimt=23.34,
                wintrc=89.20000,
                wslope=0.503750,
                wxlimt=21.00,
            ),
        ),
        # October (`seas=4`) will get interpolated from the previous and next
        # (first in this list) season.
    ],
)

# GODDARD_AMSR2_SOUTH_PARAMS = ParamsDict(
NSIDC0001_F17_SOUTH_PARAMS = ParamsDict(
    vh37_params=TbSetParams(
        # water_tie_point_set=(Tiepoint(207.6), Tiepoint(131.9)),
        # ice_tie_point_set=(Tiepoint(259.4), Tiepoint(247.3)),
        # lnline=Line(offset=-90.62, slope=1.2759),
        water_tie_point_set=(Tiepoint(201.990), Tiepoint(133.943)),
        ice_tie_point_set=(Tiepoint(259.122), Tiepoint(248.284)),
        lnline=Line(offset=-90.9384, slope=1.28239),
        # iceline=Line(offset=-40.8250, slope=1.11404),  # not yet used?
    ),
    v1937_params=TbSetParams(
        # water_tie_point_set=(Tiepoint(207.6), Tiepoint(182.7)),
        # ice_tie_point_set=(Tiepoint(259.4), Tiepoint(261.6)),
        # lnline=Line(offset=62.89, slope=0.7618),
        water_tie_point_set=(Tiepoint(201.990), Tiepoint(178.358)),
        ice_tie_point_set=(Tiepoint(259.122), Tiepoint(261.654)),
        lnline=Line(offset=61.7438, slope=0.767205),
        # iceline=Line(offset=114.825, slope=0.570622),  # not yet used?
    ),
    weather_filter_seasons=[
        # Just one season for the S. hemisphere.
        WeatherFilterParamsForSeason(
            start_month=1,
            end_month=12,
            weather_filter_params=WeatherFilterParams(
                # wintrc=85.13,
                # wslope=0.5379,
                # wxlimt=18.596,
                wintrc=93.2861,
                wslope=0.497374,
                wxlimt=16.5,
            ),
        ),
    ],
)

# TODO: Is this where we would overwrite the default values?
# BOOTSTRAP_PARAMS_INITIAL_AMSR2_NORTH = dict(
BOOTSTRAP_PARAMS_INITIAL_F17_NORTH = dict(
    # bt_wtp_v37=GODDARD_AMSR2_NORTH_PARAMS["vh37_params"].water_tie_point_set[0],
    # etc...
    bt_wtp_v37=NSIDC0001_F17_NORTH_PARAMS["vh37_params"].water_tie_point_set[0],
    bt_wtp_h37=NSIDC0001_F17_NORTH_PARAMS["vh37_params"].water_tie_point_set[1],
    bt_wtp_v19=NSIDC0001_F17_NORTH_PARAMS["v1937_params"].water_tie_point_set[1],
    bt_itp_v37=NSIDC0001_F17_NORTH_PARAMS["vh37_params"].ice_tie_point_set[0],
    bt_itp_h37=NSIDC0001_F17_NORTH_PARAMS["vh37_params"].ice_tie_point_set[1],
    bt_itp_v19=NSIDC0001_F17_NORTH_PARAMS["v1937_params"].ice_tie_point_set[1],
    vh37_lnline=NSIDC0001_F17_NORTH_PARAMS["vh37_params"].lnline,
    v1937_lnline=NSIDC0001_F17_NORTH_PARAMS["v1937_params"].lnline,
    weather_filter_seasons=NSIDC0001_F17_NORTH_PARAMS["weather_filter_seasons"],
)

# BOOTSTRAP_PARAMS_INITIAL_AMSR2_SOUTH = dict(
BOOTSTRAP_PARAMS_INITIAL_F17_SOUTH = dict(
    # bt_wtp_v37=GODDARD_AMSR2_SOUTH_PARAMS["vh37_params"].water_tie_point_set[0],
    bt_wtp_v37=NSIDC0001_F17_SOUTH_PARAMS["vh37_params"].water_tie_point_set[0],
    bt_wtp_h37=NSIDC0001_F17_SOUTH_PARAMS["vh37_params"].water_tie_point_set[1],
    bt_wtp_v19=NSIDC0001_F17_SOUTH_PARAMS["v1937_params"].water_tie_point_set[1],
    bt_itp_v37=NSIDC0001_F17_SOUTH_PARAMS["vh37_params"].ice_tie_point_set[0],
    bt_itp_h37=NSIDC0001_F17_SOUTH_PARAMS["vh37_params"].ice_tie_point_set[1],
    bt_itp_v19=NSIDC0001_F17_SOUTH_PARAMS["v1937_params"].ice_tie_point_set[1],
    vh37_lnline=NSIDC0001_F17_SOUTH_PARAMS["vh37_params"].lnline,
    v1937_lnline=NSIDC0001_F17_SOUTH_PARAMS["v1937_params"].lnline,
    weather_filter_seasons=NSIDC0001_F17_SOUTH_PARAMS["weather_filter_seasons"],
)


CDR_F17_NORTH_PARAMS = NSIDC0001_F17_NORTH_PARAMS.copy()

CDR_F17_SOUTH_PARAMS = NSIDC0001_F17_SOUTH_PARAMS.copy()


def get_F17_params(
    *,
    hemisphere: Hemisphere,
) -> BootstrapParams:
    bt_params = BootstrapParams(
        **(CDR_F17_NORTH_PARAMS if hemisphere == "north" else CDR_F17_SOUTH_PARAMS),
    )

    return bt_params


def get_F17_bootstrap_params(
    *,
    date: dt.date,
    satellite: str,
    gridid: str,
) -> dict:
    """Assign the bootstrap parameters for this date, sat, grid."""
    hemisphere = get_gridid_hemisphere(gridid)
    # TODO: Need to add a "ame" conditional here
    if satellite == "F17":
        if hemisphere == "north":
            initial_bt_params = BOOTSTRAP_PARAMS_INITIAL_F17_NORTH
        elif hemisphere == "south":
            initial_bt_params = BOOTSTRAP_PARAMS_INITIAL_F17_SOUTH
        else:
            raise ValueError(
                "Could not initialize Bootstrap params for:\n"
                f"satellite: {satellite}\n  hemisphere: {hemisphere}"
                f"\n  gridid: {gridid}"
            )
    else:
        raise ValueError(
            f"Bootstrap params not yet defined for:\n  satellite: {satellite}"
        )

    bt_params = setup_bootstrap_params_dict(
        initial_params_dict=initial_bt_params, date=date
    )

    return bt_params
