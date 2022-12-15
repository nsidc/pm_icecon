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

from pm_icecon.bt.compute_bt_via_recipe import (
    bootstrap_via_recipe, 
    get_standard_bootstrap_recipe,
)


"""
Notes:
amsr2_bootstrap() is in pm_icecon.bt.api.py
get_au_si_tbs() is in pm_icecon.fetch.au_si
"""

_initial_bt_parameter_list = (
        'wtp_v37_init', 'wtp_h37_init', 'wtp_v19_init',
        'itp_v37', 'itp_h37', 'itp_v19',
        'vh37_lnline_offset', 'vh37_lnline_slope',
        'v1937_lnline_offset', 'v1937_lnline_slope',
    )


def test_bt_via_recipe_returns_Dataset():
    """
    Test that the generation of a bootstrap code via a "recipe"
    yields an xarray Dataset
    """
    bt_recipe = get_standard_bootstrap_recipe(gridid='psn12.5', tb_source='au_si12')
    bt = bootstrap_via_recipe(recipe=bt_recipe)

    assert type(bt) == type(xr.Dataset())


def test_get_recipe_has_appropriate_keys():
    """
    Test that the generation of a bootstrap code via a "recipe"

    """
    bt_recipe = get_standard_bootstrap_recipe(gridid='psn12.5', tb_source='au_si12')
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
    bt_recipe = get_standard_bootstrap_recipe(gridid='psn12.5', tb_source='au_si12')
    bt = bootstrap_via_recipe(recipe=bt_recipe)

    initial_tb_fields = \
        ('tb_v37_in', 'tb_h37_in', 'tb_v19_in', 'tb_v22_in')

    for tb_field in initial_tb_fields:
        assert tb_field in bt.variables.keys()


def test_bt_recipe_yields_spatially_interpolated_tbs():
    """
    Test that TB fields from AI_SI12 can be acquired via the bootstrap recipe
    """
    bt_recipe = get_standard_bootstrap_recipe(gridid='psn12.5', tb_source='au_si12')
    bt = bootstrap_via_recipe(recipe=bt_recipe)

    spatially_interpolated_tb_fields = \
        ('tb_v37_si', 'tb_h37_si', 'tb_v19_si', 'tb_v22_si')

    for tb_field in spatially_interpolated_tb_fields:
        assert tb_field in bt.variables.keys()


def test_bt_recipe_yields_masks():
    """
    Test that standard masks can be loaded via the bootstrap recipe
    """
    bt_recipe = get_standard_bootstrap_recipe(gridid='psn12.5', tb_source='au_si12')
    bt = bootstrap_via_recipe(recipe=bt_recipe)

    mask_list = ('surface_mask', 'invalid_ice_mask',)

    for mask in mask_list:
        assert mask in bt.variables.keys()

    if 'n' in bt_recipe['run_parameters']['gridid']:
        assert 'pole_mask' in bt.variables.keys()


def test_bt_recipe_has_bt_parameters():
    """
    Test that standard masks can be loaded via the bootstrap recipe
    """
    bt_recipe = get_standard_bootstrap_recipe(gridid='psn12.5', tb_source='au_si12')
    for bt_parameter in _initial_bt_parameter_list:
        assert bt_parameter in bt_recipe['bootstrap_parameters'].keys()


def test_bt_recipe_yields_icecon_parameters():
    """
    Test that standard masks can be loaded via the bootstrap recipe
    """
    bt_recipe = get_standard_bootstrap_recipe(gridid='psn12.5', tb_source='au_si12')
    bt = bootstrap_via_recipe(recipe=bt_recipe)

    for bt_parameter in _initial_bt_parameter_list:
        assert bt_parameter in bt.variables['icecon_parameters'].attrs.keys()


def test_bt_recipe_yields_icecon():
    """
    Test that standard masks can be loaded via the bootstrap recipe
    """
    bt_recipe = get_standard_bootstrap_recipe(gridid='psn12.5', tb_source='au_si12')
    bt = bootstrap_via_recipe(recipe=bt_recipe)

    icecon_key = 'icecon'
    assert icecon_key in bt.variables.keys()


def test_bt_recipe_yields_same_icecon():
    """
    Test that standard masks can be loaded via the bootstrap recipe
    """
    bt_recipe = get_standard_bootstrap_recipe(gridid='psn12.5', tb_source='au_si12')
    bt = bootstrap_via_recipe(recipe=bt_recipe)
    bt_icecon_via_recipe = bt['icecon']

    bt_ds_orig_method = amsr2_bootstrap(
        date=dt.date(2020, 1, 1),
        hemisphere='north',
        resolution='12',
    )

    bt_icecon_via_orig_method = bt_ds_orig_method['conc']

    """
    ofn = 'bt_viarecipe.nc'
    bt.to_netcdf(ofn)
    print(f'Wrote: {ofn}')

    ofn = 'bt_viaoriginal.nc'
    bt_ds_orig_method.to_netcdf(ofn)
    print(f'Wrote: {ofn}')
    """

    assert np.all(bt_icecon_via_recipe.data == bt_icecon_via_orig_method.data)
    """
    assert_almost_equal(
        bt_icecon_via_recipe.data,
        bt_icecon_via_orig_method.data,
    )
    """


def test_bt_recipe_returns_mask_fields():
    """
    Test that the bootstrap dataset contains a valid TB mask
    """
    bt_recipe = get_standard_bootstrap_recipe(gridid='psn12.5', tb_source='au_si12')
    bt = bootstrap_via_recipe(recipe=bt_recipe)

    mask_fields = (
        'valid_tb_mask',
        'is_water_mask',
    )
    for mask_field in mask_fields:
        print(f'\nChecking mask field: {mask_field}')
        assert np.any(bt[mask_field])
