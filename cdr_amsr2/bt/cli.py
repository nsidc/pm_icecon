import datetime as dt
from pathlib import Path
from typing import get_args

import click
from loguru import logger

from cdr_amsr2._types import Hemisphere
from cdr_amsr2.bt.api import amsr2_bootstrap


@click.command()
@click.option(
    '-d',
    '--date',
    required=True,
    type=click.DateTime(formats=('%Y-%m-%d',)),
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
def amsr2(date: dt.date, hemisphere: Hemisphere, output_dir: Path):
    """Run the bootstrap algorithm with ASMR2 data.

    AMSRU2 brightness temperatures are fetched from AU_SI25.

    The resulting concentration field is saved to a netcdf file in the given
    `output_dir` with the form `{N|S}H_{YYYYMMDD}_py_NRT_amsr2.nc`
    """
    conc_ds = amsr2_bootstrap(
        date=date,
        hemisphere=hemisphere,
    )

    output_fn = f'{hemisphere[0].upper()}H_{date:%Y%m%d}_py_NRT_amsr2.nc'
    output_path = output_dir / output_fn
    conc_ds.to_netcdf(output_path)
    logger.info(f'Wrote AMSR2 concentration field: {output_path}')


@click.group(name='bootstrap')
def cli():
    """Run the bootstrap algorithm."""
    ...


cli.add_command(amsr2)


if __name__ == '__main__':
    cli()
