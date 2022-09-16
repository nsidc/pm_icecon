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
from pathlib import Path
from typing import get_args

import click
import pandas as pd
import xarray as xr
from loguru import logger

from cdr_amsr2._types import Hemisphere
from cdr_amsr2.bt.api import amsr2_bootstrap
from cdr_amsr2.cli.util import datetime_to_date
from cdr_amsr2.fetch.au_si import AU_SI_RESOLUTIONS
from cdr_amsr2.nt.api import amsr2_nasateam
from cdr_amsr2.util import standard_output_filename


def amsr2_cdr(
    *, date: dt.date, hemisphere: Hemisphere, resolution: AU_SI_RESOLUTIONS
) -> xr.Dataset:
    """Create a CDR-like concentration field from AMSR2 data."""
    bt_conc_ds = amsr2_bootstrap(
        date=date, hemisphere=hemisphere, resolution=resolution
    )
    nt_conc_ds = amsr2_nasateam(date=date, hemisphere=hemisphere, resolution=resolution)

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
    conc_ds.to_netcdf(output_path)
    logger.info(f'Wrote AMSR2 CDR concentration field: {output_path}')


def create_cdr_for_date_range(
    *,
    hemisphere: Hemisphere,
    start_date: dt.date,
    end_date: dt.date,
    resolution: AU_SI_RESOLUTIONS,
    output_dir: Path,
) -> None:
    for pd_timestamp in pd.date_range(start=start_date, end=end_date, freq='D'):
        make_cdr_netcdf(
            date=pd_timestamp.date(),
            hemisphere=hemisphere,
            resolution=resolution,
            output_dir=output_dir,
        )


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
    for hemisphere in get_args(Hemisphere):
        create_cdr_for_date_range(
            start_date=dt.date(2021, 1, 1),
            end_date=dt.date(2021, 12, 31),
            hemisphere=hemisphere,
            resolution='25',
            output_dir=Path('/home/vagrant/cdr_data/'),
        )
