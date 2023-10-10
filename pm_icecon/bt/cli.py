import datetime as dt
from pathlib import Path
from typing import get_args

import click
import numpy as np
from loguru import logger
from pm_tb_data.fetch.a2l1c_utils import (
    add_info_to_netcdf_file_a2l1c,
    create_equivalent_geotiff_a2l1c,
    derive_geotiff_name_a2l1c,
)
from pm_tb_data.fetch.au_si import AU_SI_RESOLUTIONS

from pm_icecon._types import Hemisphere
from pm_icecon.bt.api import a2l1c_goddard_bootstrap, amsr2_goddard_bootstrap
from pm_icecon.cli.util import datetime_to_date
from pm_icecon.util import standard_output_filename


# Click definitions for "amsr2" which uses AU25
@click.command()  # type: ignore
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
def amsr2(
    *,
    date: dt.date,
    hemisphere: Hemisphere,
    output_dir: Path,
    resolution: AU_SI_RESOLUTIONS,
):
    """Run the bootstrap algorithm with ASMR2 data.

    AMSRU2 brightness temperatures are fetched from AU_SI25.

    The resulting concentration field is saved to a netcdf file in the given
    `output_dir` with the form `{N|S}H_{YYYYMMDD}_py_NRT_amsr2.nc`
    """
    conc_ds = amsr2_goddard_bootstrap(
        date=date,
        hemisphere=hemisphere,
        resolution=resolution,
    )

    output_fn = standard_output_filename(
        hemisphere=hemisphere,
        date=date,
        sat='u2',
        resolution=f'{resolution}km',
        algorithm='bt',
    )
    output_path = output_dir / output_fn
    conc_ds.to_netcdf(output_path)
    logger.info(f'Wrote AMSR2 concentration field: {output_path}')


# Click definitions for 'a2l1c' which uses 6.25km fields derived from 0763
@click.command()  # type: ignore
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
    '-t',
    '--tb_dir',
    required=True,
    type=click.Path(
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
        path_type=Path,
    ),
)
@click.option(
    '-a',
    '--anc_dir',
    required=True,
    type=click.Path(
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
        path_type=Path,
    ),
)
@click.option(
    '-p',
    '--nctbfn_template',
    required=False,
    default='NSIDC-0763-EASE2_{hemlet}{gridres}km-GCOMW1_AMSR2-{year}{doy}-{capchan}-{tim}-SIR-PPS_XCAL-v1.1.nc',  # noqa
)
@click.option(
    '-f',
    '--timeframe',
    required=False,
    default='M',
)
def a2l1c(
    *,
    date: dt.date,
    hemisphere: Hemisphere,
    output_dir: Path,
    tb_dir: Path,
    anc_dir: Path,
    nctbfn_template: str,
    timeframe: str,
):
    """Run the bootstrap algorithm with 'a2l1c' data.

    AMSR brightness temperatures are fetched from 6.25km raw data fields
    derived from nsidc0763 (unpublished) files created from L1C fields

    The resulting concentration field is saved to a netcdf file in the given
    `output_dir`.
    """
    resolution = '6.25km'
    conc_ds = a2l1c_goddard_bootstrap(
        date=date,
        hemisphere=hemisphere,
        tb_dir=tb_dir,
        anc_dir=anc_dir,
        ncfn_template=nctbfn_template,
        timeframe=timeframe,
    )

    output_fn = standard_output_filename(
        hemisphere=hemisphere,
        date=date,
        sat='a2l1c',
        resolution=resolution,
        algorithm='bt',
        timeframe=timeframe,
    )
    output_path = output_dir / output_fn
    conc_ds.astype(np.float32).to_netcdf(output_path, encoding={'conc': {'zlib': True}})

    logger.info(f'Wrote a2l1c concentration field: {output_path}')

    # Note: the following command replaces the output file
    add_info_to_netcdf_file_a2l1c(output_path)
    logger.info(f'Re-wrote AMSR2 concentration netCDF file: {output_path}')

    # Write an equivalent geotiff file
    geotiff_output_path = derive_geotiff_name_a2l1c(output_path)
    create_equivalent_geotiff_a2l1c(output_path, geotiff_output_path)
    logger.info(f'Wrote AMSR2 concentration geotiff: {geotiff_output_path}')


@click.group(name='bootstrap')
def cli():
    """Run the bootstrap algorithm."""
    ...


cli.add_command(amsr2)
cli.add_command(a2l1c)


if __name__ == '__main__':
    cli()
