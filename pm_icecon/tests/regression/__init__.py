from pathlib import Path


# NSIDC infrastructure-specific paths:
NSIDC_NFS_SHARE_DIR = Path("/share/apps/amsr2-cdr")

# Contains regression data, ancillary data files (masks), etc.
# TODO: move ancillary files out of `cdr_testdata` dir? On the VMs, the
# ancillary nasateam files in cdr_testdata are also in the `NSIDC_NFS_SHARE_DIR`
CDR_TESTDATA_DIR = NSIDC_NFS_SHARE_DIR / "cdr_testdata"
