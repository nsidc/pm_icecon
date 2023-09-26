"""Locate tb files on disk and return xarray objects representing the data.

E.g,

```
import datetime as dt


tbs = get_a2l1c_625_tbs(
    base_dir=Path('/data/amsr2_subsets'),
    date=dt.date(2022, 1, 15),
    hemisphere='north',
)
```

Adapted from au_si25.py
"""
import datetime as dt
import re
import warnings
from pathlib import Path

import numpy as np
import xarray as xr
from netCDF4 import Dataset

from pm_icecon._types import Hemisphere


def _get_a2l1c_625_data_fields_nc(
    *,
    base_dir: Path,
    date: dt.date,
    hemisphere: Hemisphere,
    verbose=True,
    tbfn_template='NSIDC-0763-EASE2_{hemlet}{gridres}km-GCOMW1_AMSR2-{year}{doy}-{capchan}-{tim}-SIR-PPS_XCAL-v1.1.nc': str,  # noqa
    timeframe: str,
) -> xr.Dataset:
    """Find raw binary files used for 6.25km NH from AMSR2 L1C (NSIDC-0763).

    Returns an xarray dataset of the variables.
    """
    year = date.strftime('%Y')  # year, 4 char string
    doy = date.strftime('%j')  # day-of-year, 3 char string
    if timeframe in ('M', 'E'):
        tim = timeframe
    else:
        raise ValueError(f'Unrecognized timeframe: {timeframe}')

    tbs = {}
    chans = ('18v', '23v', '36h', '36v')
    for chan in chans:
        if int(chan[:2]) < 30:
            # native SIR grid is 6.25km
            gridres = '6.25'
        else:
            # native SIR grid is 3.125km
            gridres = '3.125'
        tbfn = tbfn_.format(
            hemlet=hemisphere[0].upper(),
            gridres=gridres,
            year=year,
            doy=doy,
            capchan=chan.upper(),
            tim=tim,
        )
        full_path = base_dir / Path(tbfn)
        ds = Dataset(full_path, 'r')
        tb_data = np.array(ds.variables['TB']).squeeze()

        # Need to convert 0 to nan
        tb_data[tb_data == 0] = np.nan

        # Convert 3.125km to 6.25 grid if needed
        if int(chan[:2]) > 30:
            dim, _ = tb_data.shape
            new_dim = dim // 2
            tb_grouped = tb_data.reshape(-1, 2, new_dim, 2)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=RuntimeWarning)
                tb_data = np.nanmean(tb_grouped, (-1, -3))

        # Only use a subset of the full hemisphere
        tbs[chan] = tb_data[600 : 600 + 1680, 600 : 600 + 1680]

    ds = xr.Dataset(
        data_vars=dict(
            v18=(['x', 'y'], tbs['18v']),
            v23=(['x', 'y'], tbs['23v']),
            h36=(['x', 'y'], tbs['36h']),
            v36=(['x', 'y'], tbs['36v']),
        ),
        attrs=dict(description=f'a2l1c tb fields for a2l1c BT for {date}'),
    )

    return ds


def _get_a2l1c_625_data_fields(
    *,
    base_dir: Path,
    date: dt.date,
    hemisphere: Hemisphere,
    verbose=False,
    timeframe,
) -> xr.Dataset:
    """Find raw binary files used for 6.25km NH from AMSR2 L1C (NSIDC-0763).

    Returns an xarray dataset of the variables.
    """
    chans = ('18v', '23v', '36h', '36v')
    dim = 1680
    if timeframe == 'M':
        tim = 'am'
    elif timeframe == 'E':
        tim = 'pm'
    else:
        raise ValueError(f'timeframe not M or E: {timeframe}')
    ymdstr = date.strftime('%Y%m%d')
    if verbose:
        print('Assuming:')
        print('  variables are 6.25km grid,')
        print('  a 1680x1680 subset of E2N')
        print('Set:')
        print(f'  chans: {chans}')
        print(f'    dim: {dim}')
        print(f'    tim: {tim}')
        print(f'    ymd: {ymdstr}')
        print(f'    tim: {tim}')

    fn = {}
    tbs = {}

    for chan in chans:
        fn[chan] = f'{base_dir}/tb_a2im_sir_{chan}_{tim}_e2n6.25_{ymdstr}.dat'
        tbs[chan] = np.divide(
            np.fromfile(fn[chan], dtype=np.int16).reshape(dim, dim), 100.0
        )

    ds = xr.Dataset(
        data_vars=dict(
            v18=(['x', 'y'], tbs['18v']),
            v23=(['x', 'y'], tbs['23v']),
            h36=(['x', 'y'], tbs['36h']),
            v36=(['x', 'y'], tbs['36v']),
        ),
        attrs=dict(description=f'a2l1c tb fields for CDR BT for {date}'),
    )

    return ds


def _normalize_a2l1c_625_tbs(
    data_fields: xr.Dataset,
) -> xr.Dataset:
    """Normalize the given a2l1c Tbs.

    renames the original -- from filename -- freqpol to polfreq,
      eg '18v' becomes 'v18'

    Reminder: chans = ('18v', '23v', '36h', '36v')
    """
    var_pattern = re.compile(r'(?P<polarization>h|v)(?P<channel>\d{2})')

    tb_data_mapping = {}
    for var in data_fields.keys():
        # print(f'checking xr var: {var}')
        if match := var_pattern.match(str(var)):
            # print('  matches!')
            tb_data_mapping[
                f"{match.group('polarization').lower()}{match.group('channel')}"
            ] = data_fields[var]

    normalized = xr.Dataset(tb_data_mapping)

    return normalized


def get_a2l1c_625_tbs(
    *,
    base_dir: Path,
    date: dt.date,
    hemisphere: Hemisphere,
    ncfn_template,
    timeframe,
) -> xr.Dataset:
    """Return CETB Tbs for the given date and hemisphere as an xr dataset."""
    try:
        # First, try to load pre-extracted raw binary files
        data_fields = _get_a2l1c_625_data_fields(
            base_dir=base_dir,
            date=date,
            hemisphere=hemisphere,
            timeframe=timeframe,
        )
    except FileNotFoundError:
        # If no bin files, attempt to load from 0763 netcdf files
        try:
            data_fields = _get_a2l1c_625_data_fields_nc(
                base_dir=base_dir,
                date=date,
                hemisphere=hemisphere,
                tbfn_template=ncfn_template,
                timeframe=timeframe,  # noqa
            )
        except FileNotFoundError:
            raise SystemExit(f'Could not find a2l1c input files in {base_dir}')

    tb_data = _normalize_a2l1c_625_tbs(data_fields)

    return tb_data
