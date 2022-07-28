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
from pathlib import Path

import numpy as np
import xarray as xr

from cdr_amsr2._types import Hemisphere


def _get_a2l1c_625_data_fields(
    *,
    base_dir: Path,
    date: dt.date,
    hemisphere: Hemisphere,
) -> xr.Dataset:
    """Find raw binary files used for 6.25km NH from AMSR2 L1C (NSIDC-0763).

    Returns an xarray dataset of the variables.
    """
    chans = ('18v', '23v', '36h', '36v')
    dim = 1680
    tim = 'am'
    ymdstr = date.strftime('%Y%m%d')
    print('Assuming:')
    print('  variables are 6.25km grid,')
    print('  a 1680x1680 subset of E2N')
    print('  time is "am"')
    print('Set:')
    print(f'  chans: {chans}')
    print(f'    dim: {dim}')
    print(f'    tim: {tim}')
    print(f'    ymd: {ymdstr}')

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
        print(f'checking xr var: {var}')
        if match := var_pattern.match(var):
            print('  matches!')
            tb_data_mapping[
                f"{match.group('polarization').lower()}{match.group('channel')}"
            ] = data_fields[var]

    normalized = xr.Dataset(tb_data_mapping)

    return normalized


# def get_au_si25_tbs(
def get_a2l1c_625_tbs(
    *, base_dir: Path, date: dt.date, hemisphere: Hemisphere
) -> xr.Dataset:
    """Return AU_SI25 Tbs for the given date and hemisphere as an xr dataset."""
    data_fields = _get_a2l1c_625_data_fields(
        base_dir=base_dir, date=date, hemisphere=hemisphere
    )
    tb_data = _normalize_a2l1c_625_tbs(data_fields)

    return tb_data
