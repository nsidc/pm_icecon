import datetime as dt
from pathlib import Path

import numpy as np
import xarray as xr
from numpy.testing import assert_almost_equal
from numpy.typing import NDArray

import pm_icecon.bt.compute_bt_ic as bt
from pm_icecon._types import Hemisphere
from pm_icecon.bt.api import amsr2_bootstrap
from pm_icecon.bt.compute_bt_ic import xfer_class_tbs
from pm_icecon.bt.masks import get_ps_invalid_ice_mask
from pm_icecon.bt.params.goddard_class import SSMIS_NORTH_PARAMS
from pm_icecon.config.models.bt import BootstrapParams
from pm_icecon.constants import CDR_TESTDATA_DIR
from pm_icecon.fetch.au_si import AU_SI_RESOLUTIONS
from pm_icecon.interpolation import spatial_interp_tbs
from pm_icecon.masks import get_ps_land_mask, get_ps_pole_hole_mask

from pm_icecon.bt.compute_bt_via_recipe import bootstrap_via_recipe, get_standard_bootstrap_recipe


"""
Notes:
amsr2_bootstrap() is in pm_icecon.bt.api.py
get_au_si_tbs() is in pm_icecon.fetch.au_si
"""

def test_bt_via_recipe_returns_Dataset():
    """
    Test that the generation of a bootstrap code via a "recipe"
    yields an xarray Dataset
    """
    bt_recipe = get_standard_bootstrap_recipe()
    bt = bootstrap_via_recipe(recipe=bt_recipe)

    assert type(bt) == type(xr.Dataset())


def test_get_recipe_has_appropriate_keys():
    """
    Test that the generation of a bootstrap code via a "recipe"

    """
    bt_recipe = get_standard_bootstrap_recipe()
    assert type(bt_recipe) == dict

    recipe_elements = \
        ('run_parameters', 'tb_parameters',
         'bootstrap_parameters', 'ancillary_sources',
        )
    for recipe_element in recipe_elements:
        assert recipe_element in bt_recipe.keys()


def test_bt_recipe_yields_ausi12_tbs():
    """
    Test that TB fields from AI_SI12 can be acquired via the bootstrap recipe
    """
    bt_recipe = get_standard_bootstrap_recipe()
    bt = bootstrap_via_recipe(recipe=bt_recipe)

    initial_tb_fields = \
        ('tb_v37_in', 'tb_h37_in', 'tb_v19_in', 'tb_v22_in')

    for tb_field in initial_tb_fields:
        assert tb_field in bt.variables.keys()


def test_bt_recipe_yields_spatially_interpolated_tbs():
    """
    Test that TB fields from AI_SI12 can be acquired via the bootstrap recipe
    """
    bt_recipe = get_standard_bootstrap_recipe()
    bt = bootstrap_via_recipe(recipe=bt_recipe)

    spatially_interpolated_tb_fields = \
        ('tb_v37_si', 'tb_h37_si', 'tb_v19_si', 'tb_v22_si')

    for tb_field in spatially_interpolated_tb_fields:
        assert tb_field in bt.variables.keys()


def test_bt_recipe_yields_masks():
    """
    Test that standard masks can be loaded via the bootstrap recipe
    """
    bt_recipe = get_standard_bootstrap_recipe()
    bt = bootstrap_via_recipe(recipe=bt_recipe)

    mask_list = ('surface_mask', 'invalid_ice_mask',)

    for mask in mask_list:
        assert mask in bt.variables.keys()

    if 'n' in bt_recipe['run_parameters']['gridid']:
        assert 'pole_mask' in bt.variables.keys()


def test_bt_recipe_yields_icecon_parameters():
    """
    Test that standard masks can be loaded via the bootstrap recipe
    """
    bt_recipe = get_standard_bootstrap_recipe()
    bt = bootstrap_via_recipe(recipe=bt_recipe)

    preset_keys = ('vh37_params', 'v1937_params', 'weather_filter_seasons')

    print(f"preset attr keys:\n{bt.variables['icecon_parameters'].attrs.keys()}")
    for preset_key in preset_keys:
        print(f"Checking for {preset_key} in {bt.variables['icecon_parameters'].attrs.keys()}")
        assert preset_key in bt.variables['icecon_parameters'].attrs.keys()


def test_bt_recipe_yields_icecon():
    """
    Test that standard masks can be loaded via the bootstrap recipe
    """
    bt_recipe = get_standard_bootstrap_recipe()
    bt = bootstrap_via_recipe(recipe=bt_recipe)

    icecon_key = 'icecon'
    assert icecon_key in bt.variables.keys()


def test_bt_recipe_yields_same_icecon():
    """
    Test that standard masks can be loaded via the bootstrap recipe
    """
    bt_recipe = get_standard_bootstrap_recipe()
    bt = bootstrap_via_recipe(recipe=bt_recipe)
    bt_icecon_via_recipe = bt['icecon']

    bt_ds_orig_method = amsr2_bootstrap(
        date=dt.date(2020, 1, 1),
        hemisphere='north',
        resolution='12',
        # resolution='25',
    )

    bt_icecon_via_orig_method = bt_ds_orig_method['conc']

    ofn = 'bt_viarecipe.nc'
    bt.to_netcdf(ofn)
    print(f'Wrote: {ofn}')

    ofn = 'bt_viaoriginal.nc'
    bt_ds_orig_method.to_netcdf(ofn)
    print(f'Wrote: {ofn}')

    assert_almost_equal(
        bt_icecon_via_recipe.data,
        bt_icecon_via_orig_method.data,
    )
    print(f'Two icecons are "almost equal"!')


'''
def test_bt_amsr2_regression():
    """Regression test for BT AMSR2 outputs.

    Compare output from bt algorithm for 2020-01-01 and 2022-05-04 against
    regression data.

    Scott Stewart manually examined the regression data and determined it looks
    good. These fields may need to be updated as we make tweaks to the
    algorithm.
    """
    for date in (dt.date(2020, 1, 1), dt.date(2022, 5, 4)):
        actual_ds = amsr2_bootstrap(
            date=date,
            hemisphere='north',
            resolution='25',
        )
        filename = f'NH_{date:%Y%m%d}_py_NRT_amsr2.nc'
        regression_ds = xr.open_dataset(
            CDR_TESTDATA_DIR / 'bt_amsru_regression' / filename
        )
        assert_almost_equal(
            regression_ds.conc.data,
            actual_ds.conc.data,
            decimal=1,
        )


def _original_f18_example() -> xr.Dataset:
    """Return concentration field example for f18_20180217.

    This example data does not perfectly match the outputs given by Goddard's
    code, but it is very close. A total of 4 cells differ 1.

    ```
    >>> exact[not_eq]
    array([984, 991, 975, 830], dtype=int16)
    >>> not_eq = exact != not_exact
    >>> not_exact[not_eq]
    array([983, 992, 974, 829], dtype=int16)
    ```

    the exact grid produced by the fortran code is in
    `CDR_TESTDATA / 'bt_goddard_orig_output/NH_20180217_SB2_NRT_f18.ic'`.
    """
    resolution: AU_SI_RESOLUTIONS = '25'
    date = dt.date(2018, 2, 17)
    hemisphere: Hemisphere = 'north'
    params = BootstrapParams(
        land_mask=get_ps_land_mask(hemisphere=hemisphere, resolution=resolution),
        pole_mask=get_ps_pole_hole_mask(resolution=resolution),
        invalid_ice_mask=get_ps_invalid_ice_mask(
            hemisphere=hemisphere,
            date=date,
            resolution=resolution,  # type: ignore[arg-type]
        ),
        **SSMIS_NORTH_PARAMS,
    )

    otbs: dict[str, NDArray[np.float32]] = {}

    orig_input_tbs_dir = CDR_TESTDATA_DIR / 'bt_goddard_orig_input_tbs/'
    raw_fns = {
        'v19': 'tb_f18_20180217_nrt_n19v.bin',
        'h37': 'tb_f18_20180217_nrt_n37h.bin',
        'v37': 'tb_f18_20180217_nrt_n37v.bin',
        'v22': 'tb_f18_20180217_nrt_n22v.bin',
    }

    def _read_tb_field(tbfn: Path) -> NDArray[np.float32]:
        # Read int16 scaled by 10 and return float32 unscaled
        raw = np.fromfile(tbfn, dtype=np.int16).reshape(448, 304)

        return bt.fdiv(raw.astype(np.float32), 10)

    for tb in ('v19', 'h37', 'v37', 'v22'):
        otbs[tb] = _read_tb_field(
            (
                orig_input_tbs_dir / raw_fns[tb]  # type: ignore [literal-required]
            ).resolve()
        )

    conc_ds = bt.bootstrap(
        # Apply expected transformation for F18 CLASS data.
        **xfer_class_tbs(  # type: ignore[arg-type]
            tb_v37=spatial_interp_tbs(otbs['v37']),
            tb_h37=spatial_interp_tbs(otbs['h37']),
            tb_v19=spatial_interp_tbs(otbs['v19']),
            tb_v22=spatial_interp_tbs(otbs['v22']),
            sat='f18',
        ),
        params=params,
        date=date,
        hemisphere=hemisphere,
    )

    return conc_ds


def test_bt_f18_regression():
    """Regressi5on test for BT F18 output."""
    actual_ds = _original_f18_example()
    regression_ds = xr.open_dataset(
        CDR_TESTDATA_DIR / 'bt_f18_regression/NH_20180217_NRT_f18_regression.nc',
    )

    assert_almost_equal(
        regression_ds.conc.data,
        actual_ds.conc.data,
        decimal=1,
    )
'''
