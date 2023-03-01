"""Create a 'simplified' CDR for comparison purposes.

Temporary code for simulating the sea ice CDR for comparison and demonstration purposes.

The CDR algorithm is:

* spatial interpolation on input Tbs. The NT and BT API for AMSR2 currently have
  this implemented.
* Choose bootstrap unless nasateam is larger where bootstrap has ice.

Eventually this code will be removed/migrated to the sea ice cdr project. This
project should be primarily responsible for generating concentration fields from
input Tbs.
"""
import datetime as dt
import traceback
from pathlib import Path
from typing import get_args

import click
import numpy as np
import numpy.typing as npt
import xarray as xr
from loguru import logger

import pm_icecon.bt.compute_bt_ic as bt
import pm_icecon.nt.compute_nt_ic as nt
from pm_icecon._types import Hemisphere
from pm_icecon.bt.api import amsr2_goddard_bootstrap
from pm_icecon.cli.util import datetime_to_date
from pm_icecon.config.models.bt import BootstrapParams
from pm_icecon.constants import CDR_DATA_DIR
from pm_icecon.fetch.au_si import AU_SI_RESOLUTIONS
from pm_icecon.nt._types import (
    NasateamCoefficients,
    NasateamGradientRatioThresholds,
    NasateamRatio,
)
from pm_icecon.nt.api import amsr2_goddard_nasateam
from pm_icecon.nt.tiepoints import NasateamTiePoints
from pm_icecon.util import date_range, standard_output_filename


def cdr(
    date: dt.date,
    tb_h19: npt.NDArray,
    tb_v37: npt.NDArray,
    tb_h37: npt.NDArray,
    tb_v19: npt.NDArray,
    tb_v22: npt.NDArray,
    bt_params: BootstrapParams,
    nt_tiepoints: NasateamTiePoints,
    nt_gradient_thresholds: NasateamGradientRatioThresholds,
    nt_invalid_ice_mask: npt.NDArray[np.bool_],
    nt_minic: npt.NDArray,
    shoremap: npt.NDArray,
    missing_flag_value,
    land_flag_value,
) -> xr.Dataset:
    """Run the CDR algorithm."""
    # First, get bootstrap conc.
    bt_tb_mask = bt.tb_data_mask(
        tbs=(
            tb_v37,
            tb_h37,
            tb_v19,
            tb_v22,
        ),
        min_tb=bt_params.mintb,
        max_tb=bt_params.maxtb,
    )

    bt_weather_mask = bt.get_weather_mask(
        v37=tb_v37,
        h37=tb_h37,
        v22=tb_v22,
        v19=tb_v19,
        land_mask=bt_params.land_mask,
        tb_mask=bt_tb_mask,
        ln1=bt_params.vh37_params.lnline,
        date=date,
        weather_filter_seasons=bt_params.weather_filter_seasons,
    )
    bt_conc = bt.bootstrap_for_cdr(
        tb_v37=tb_v37,
        tb_h37=tb_h37,
        tb_v19=tb_v19,
        params=bt_params,
        tb_mask=bt_tb_mask,
        weather_mask=bt_weather_mask,
    )

    # Next, get nasateam conc.
    nt_pr_1919 = nt.compute_ratio(tb_v19, tb_h19)
    nt_gr_3719 = nt.compute_ratio(tb_v37, tb_v19)
    nt_conc = nt.calc_nasateam_conc(
        pr_1919=nt_pr_1919,
        gr_3719=nt_gr_3719,
        tiepoints=nt_tiepoints,
    )

    # Now calculate CDR SIC
    is_bt_seaice = (bt_conc > 0) & (bt_conc <= 100)
    use_nt_values = (nt_conc > bt_conc) & is_bt_seaice
    cdr_conc = bt_conc.copy()
    cdr_conc[use_nt_values] = nt_conc[use_nt_values]

    # Apply masks
    # Get Nasateam weather filter
    nt_gr_2219 = nt.compute_ratio(tb_v22, tb_v19)
    nt_weather_mask = nt.get_weather_filter_mask(
        gr_2219=nt_gr_2219,
        gr_3719=nt_gr_3719,
        gr_2219_threshold=nt_gradient_thresholds['2219'],
        gr_3719_threshold=nt_gradient_thresholds['3719'],
    )
    # Apply weather filters and invalid ice masks
    # TODO: can we just use a single invalid ice mask?
    set_to_zero_sic = (
        nt_weather_mask
        & bt_weather_mask
        & nt_invalid_ice_mask
        & bt_params.invalid_ice_mask
        & bt_tb_mask
    )
    cdr_conc[set_to_zero_sic] = 0

    # Apply land spillover corrections
    # nasateam first:
    cdr_conc = nt.apply_nt_spillover(conc=cdr_conc, shoremap=shoremap, minic=nt_minic)
    # then bootstrap:
    # TODO: the bootstrap land spillover routine assumes that flag values are
    # already set.
    cdr_conc = bt.coastal_fix(
        conc=cdr_conc,
        missing_flag_value=missing_flag_value,
        land_flag_value=land_flag_value,
        minic=bt_params.minic,
    )

    # Apply flag values
    ...

    # Return CDR.
    ...


def amsr2_cdr(
    *, date: dt.date, hemisphere: Hemisphere, resolution: AU_SI_RESOLUTIONS
) -> xr.Dataset:
    """Create a CDR-like concentration field from AMSR2 data."""
    bt_conc_ds = amsr2_goddard_bootstrap(
        date=date, hemisphere=hemisphere, resolution=resolution
    )
    nt_conc_ds = amsr2_goddard_nasateam(
        date=date, hemisphere=hemisphere, resolution=resolution
    )

    bt_conc = bt_conc_ds.conc.data
    nt_conc = nt_conc_ds.conc.data
    is_bt_seaice = (bt_conc > 0) & (bt_conc <= 100)
    use_nt_values = (nt_conc > bt_conc) & is_bt_seaice

    cdr_conc_ds = bt_conc_ds.where(~use_nt_values, nt_conc_ds)

    return cdr_conc_ds


def make_cdr_netcdf(
    *,
    date: dt.date,
    hemisphere: Hemisphere,
    resolution: AU_SI_RESOLUTIONS,
    output_dir: Path,
) -> None:
    logger.info(f'Creating CDR for {date=}, {hemisphere=}, {resolution=}')
    conc_ds = amsr2_cdr(
        date=date,
        hemisphere=hemisphere,
        resolution=resolution,
    )

    output_fn = standard_output_filename(
        hemisphere=hemisphere,
        date=date,
        sat='u2',
        algorithm='cdr',
        resolution=f'{resolution}km',
    )
    output_path = output_dir / output_fn
    conc_ds.to_netcdf(
        output_path,
        encoding={'conc': {'zlib': True}},
    )
    logger.info(f'Wrote AMSR2 CDR concentration field: {output_path}')


def create_cdr_for_date_range(
    *,
    hemisphere: Hemisphere,
    start_date: dt.date,
    end_date: dt.date,
    resolution: AU_SI_RESOLUTIONS,
    output_dir: Path,
) -> None:
    for date in date_range(start_date=start_date, end_date=end_date):
        try:
            make_cdr_netcdf(
                date=date,
                hemisphere=hemisphere,
                resolution=resolution,
                output_dir=output_dir,
            )
        except Exception:
            logger.error(
                f'Failed to create NetCDF for {hemisphere=}, {date=}, {resolution=}.'
            )
            err_filename = standard_output_filename(
                hemisphere=hemisphere,
                date=date,
                sat='u2',
                algorithm='cdr',
                resolution=f'{resolution}km',
            )
            err_filename += '.error'
            logger.info(f'Writing error info to {err_filename}')
            with open(output_dir / err_filename, 'w') as f:
                traceback.print_exc(file=f)


@click.command(name='cdr')
@click.option(
    '-d',
    '--date',
    required=True,
    type=click.DateTime(formats=('%Y-%m-%d',)),
    callback=datetime_to_date,
)
@click.option(
    '-h',
    '--hemisphere',
    required=True,
    type=click.Choice(get_args(Hemisphere)),
)
@click.option(
    '-o',
    '--output-dir',
    required=True,
    type=click.Path(
        exists=True,
        file_okay=False,
        dir_okay=True,
        writable=True,
        resolve_path=True,
        path_type=Path,
    ),
)
@click.option(
    '-r',
    '--resolution',
    required=True,
    type=click.Choice(get_args(AU_SI_RESOLUTIONS)),
)
def cli(
    *,
    date: dt.date,
    hemisphere: Hemisphere,
    output_dir: Path,
    resolution: AU_SI_RESOLUTIONS,
) -> None:
    """Run the CDR algorithm with AMSR2 data."""
    create_cdr_for_date_range(
        start_date=date,
        end_date=date,
        hemisphere=hemisphere,
        resolution=resolution,
        output_dir=output_dir,
    )


if __name__ == '__main__':
    # vvvv MODIFY THESE PARAMETERS AS NEEDED vvvv
    start_date = dt.date(2012, 7, 2)
    end_date = dt.date(2021, 2, 11)
    resolution: AU_SI_RESOLUTIONS = '12'
    output_dir = CDR_DATA_DIR
    # ^^^^ MODIFY THESE PARAMETERS AS NEEDED ^^^^
    for hemisphere in get_args(Hemisphere):
        create_cdr_for_date_range(
            start_date=start_date,
            end_date=end_date,
            hemisphere=hemisphere,
            resolution=resolution,
            output_dir=output_dir,
        )
