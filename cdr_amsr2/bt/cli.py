import datetime as dt
from pathlib import Path
from typing import get_args

import click
from loguru import logger

from cdr_amsr2._types import Hemisphere
from cdr_amsr2.bt.api import a2l1c_bootstrap, amsr2_bootstrap


def _datetime_to_date(_ctx, _param, value: dt.datetime) -> dt.date:
    """Click callback that takes a `dt.datetime` and returns `dt.date`."""
    return value.date()


# Click definitions for "amsr2" which uses AU25
@click.command()
@click.option(
    '-d',
    '--date',
    required=True,
    type=click.DateTime(formats=('%Y-%m-%d',)),
    callback=_datetime_to_date,
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
def amsr2(*, date: dt.date, hemisphere: Hemisphere, output_dir: Path):
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


# Click definitions for 'a2l1c' which uses 6.25km fields derived from 0763
@click.command()
@click.option(
    '-d',
    '--date',
    required=True,
    type=click.DateTime(formats=('%Y-%m-%d',)),
    callback=_datetime_to_date,
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
def a2l1c(*, date: dt.date, hemisphere: Hemisphere, output_dir: Path):
    """Run the bootstrap algorithm with 'a2l1c' data.

    AMSR brightness temperatures are fetched from 6.25km raw data fields
    derived from nsidc0763 (unpublished) files created from L1C fields

    The resulting concentration field is saved to a netcdf file in the given
    `output_dir` with the form `{N|S}H_{YYYYMMDD}_py_NRT_amsr2.nc`
    """
    conc_ds = a2l1c_bootstrap(
        date=date,
        hemisphere=hemisphere,
    )

    output_fn = f'{hemisphere[0].upper()}H_{date:%Y%m%d}_py_NRT_a2l1c.nc'
    output_path = output_dir / output_fn
    conc_ds.to_netcdf(output_path)
    logger.info(f'Wrote a2l1c concentration field: {output_path}')


@click.group(name='bootstrap')
def cli():
    """Run the bootstrap algorithm."""
    ...


cli.add_command(amsr2)
cli.add_command(a2l1c)


if __name__ == '__main__':
    cli()
