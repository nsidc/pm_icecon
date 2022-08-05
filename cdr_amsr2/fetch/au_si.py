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
from typing import Literal

import xarray as xr

from cdr_amsr2._types import Hemisphere

AU_SI_RESOLUTIONS = Literal['25', '12']


def _get_au_si_data_fields(
    *,
    base_dir: Path,
    date: dt.date,
    hemisphere: Hemisphere,
    resolution: AU_SI_RESOLUTIONS,
) -> xr.Dataset:
    """Find a AU_SI* granule on disk and return the data fields as an xr ds.

    Returns an xr dataset of teh variables contained in the
    `HDFEOS/GRIDS/{N|S}pPolarGrid25km/Data Fields` group.
    """
    results = tuple(
        base_dir.glob(f'**/AMSR_U2_L3_SeaIce{resolution}km_*_{date:%Y%m%d}.he5')
    )

    if len(results) != 1:
        raise FileNotFoundError(
            f'Expected to find 1 granule for AU_SI{resolution} for {date:%Y-%m-%d}.'
            f' Found {len(results)}.'
        )

    granule_fp = results[0]
    ds = xr.open_dataset(
        granule_fp,
        group=(
            f'HDFEOS/GRIDS'
            f'/{hemisphere[0].upper()}pPolarGrid{resolution}km'
            '/Data Fields'
        ),
    )

    return ds


def _normalize_au_si_tbs(
    data_fields: xr.Dataset,
    resolution: AU_SI_RESOLUTIONS,
) -> xr.Dataset:
    """Normalize the given AU_SI* Tbs.

    Currently only returns daily average channels.

    Filters out variables that are not Tbs and renames Tbs to the 'standard'
    {channel}{polarization} name. E.g., `SI_25km_NH_06H_DAY` becomes `h06`
    """
    var_pattern = re.compile(
        f'SI_{resolution}km_' r'(N|S)H_(?P<channel>\d{2})(?P<polarization>H|V)_DAY'
    )

    tb_data_mapping = {}
    for var in data_fields.keys():
        if match := var_pattern.match(str(var)):
            tb_data_mapping[
                f"{match.group('polarization').lower()}{match.group('channel')}"
            ] = data_fields[var]

    normalized = xr.Dataset(tb_data_mapping)

    return normalized


def get_au_si_tbs(
    *,
    date: dt.date,
    hemisphere: Hemisphere,
    resolution: AU_SI_RESOLUTIONS,
) -> xr.Dataset:
    data_fields = _get_au_si_data_fields(
        base_dir=Path(f'/ecs/DP1/AMSA/AU_SI{resolution}.001/'),
        date=date,
        hemisphere=hemisphere,
        resolution=resolution,
    )
    tb_data = _normalize_au_si_tbs(data_fields, resolution=resolution)

    return tb_data
