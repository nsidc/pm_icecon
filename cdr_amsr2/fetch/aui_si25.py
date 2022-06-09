"""Locate tb files on disk and return xarray objects representing the data.

E.g,

```
import datetime as dt


tbs = get_au_si25_tbs(
    base_dir=Path('/ecs/DP1/AMSA/AU_SI25.001/'),
    date=dt.date(2022, 6, 8),
    hemisphere='north',
)
```
"""
import datetime as dt
import re
from pathlib import Path

import xarray as xr

from cdr_amsr2._types import Hemisphere


def _get_au_si25_data_fields(
    *, base_dir: Path, date: dt.date, hemisphere: Hemisphere
) -> xr.Dataset:
    """Find a AU_SI25 granule on disk and return the data fields as an xr ds.

    Returns an xr dataset of teh variables contained in the
    `HDFEOS/GRIDS/{N|S}pPolarGrid25km/Data Fields` group.
    """
    results = tuple(base_dir.glob(f'**/AMSR_U2_L3_SeaIce25km_*_{date:%Y%m%d}.he5'))

    if len(results) != 1:
        raise FileNotFoundError(
            'Expected to find 1 granule for AU_SI25 for {date:%Y%m%d}.'
            ' Found {len(results)}.'
        )

    granule_fp = results[0]
    ds = xr.open_dataset(
        granule_fp,
        group=f'HDFEOS/GRIDS/{hemisphere[0].upper()}pPolarGrid25km/Data Fields',
    )

    return ds


def _normalize_au_si25_tbs(
    data_fields: xr.Dataset,
) -> xr.Dataset:
    """Normalize the given AU_SI25 Tbs.

    Currently only returns daily average channels.

    Filters out variables that are not Tbs and renames Tbs to the 'standard'
    {channel}{polarization} name. E.g., `SI_25km_NH_06H_DAY` becomes `h06`
    """
    var_pattern = re.compile(
        r'SI_25km_(N|S)H_(?P<channel>\d{2})(?P<polarization>H|V)_DAY'
    )

    tb_data_mapping = {}
    for var in data_fields.keys():
        if match := var_pattern.match(var):
            tb_data_mapping[
                f"{match.group('polarization').lower()}{match.group('channel')}"
            ] = data_fields[var]

    normalized = xr.Dataset(tb_data_mapping)

    return normalized


def get_au_si25_tbs(
    *, base_dir: Path, date: dt.date, hemisphere: Hemisphere
) -> xr.Dataset:
    """Return AU_SI25 Tbs for the given date and hemisphere as an xr dataset."""
    data_fields = _get_au_si25_data_fields(
        base_dir=base_dir, date=date, hemisphere=hemisphere
    )
    tb_data = _normalize_au_si25_tbs(data_fields)

    return tb_data
