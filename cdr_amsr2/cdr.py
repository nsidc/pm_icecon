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

import xarray as xr

from cdr_amsr2._types import Hemisphere
from cdr_amsr2.bt.api import amsr2_bootstrap
from cdr_amsr2.nt.api import amsr2_nasateam
from cdr_amsr2.fetch.au_si import AU_SI_RESOLUTIONS


def amsr2_cdr(*, date: dt.date, hemisphere: Hemisphere, resolution: AU_SI_RESOLUTIONS) -> xr.Dataset:
    """Create a CDR-like concentration field from AMSR2 data."""
    bt_conc_ds = amsr2_bootstrap(date=date, hemisphere=hemisphere, resolution=resolution)
    nt_conc_ds = amsr2_nasateam(date=date, hemisphere=hemisphere, resolution=resolution)

    bt_conc = bt_conc_ds.conc.data
    nt_conc = nt_conc_ds.conc.data
    is_bt_seaice = (bt_conc > 0) & (bt_conc <= 100)
    use_nt_values = (nt_conc > bt_conc) & is_bt_seaice

    cdr_conc_ds = bt_conc_ds.where(~use_nt_values, nt_conc_ds)

    return cdr_conc_ds
