from pathlib import Path

from pm_icecon.config.models import FlagValues

DEFAULT_FLAG_VALUES = FlagValues()

# NSIDC infrastructure-specific paths:
NSIDC_NFS_SHARE_DIR = Path('/share/apps/amsr2-cdr')
# Produced CDR data files go here.
CDR_DATA_DIR = NSIDC_NFS_SHARE_DIR / 'cdr_data'
BOOTSTRAP_MASKS_DIR = NSIDC_NFS_SHARE_DIR / 'bootstrap_masks'
# Contains regression data, ancillary data files (masks), etc.
# TODO: move ancillary files out of `cdr_testdata` dir? On the VMs, the
# ancillary nasateam files in cdr_testdata are also in the `NSIDC_NFS_SHARE_DIR`
CDR_TESTDATA_DIR = NSIDC_NFS_SHARE_DIR / 'cdr_testdata'
BT_GODDARD_ANCILLARY_DIR = CDR_TESTDATA_DIR / 'bt_goddard_ANCILLARY'

NASATEAM2_ANCILLARY_DIR = CDR_DATA_DIR / 'nasateam2_ancillary'
