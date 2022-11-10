from pathlib import Path

from cdr_amsr2.config.models import FlagValues

PACKAGE_DIR = Path(__file__).parent

DEFAULT_FLAG_VALUES = FlagValues()

BT_GODDARD_ANCILLARY_DIR = Path(
    '/share/apps/amsr2-cdr/cdr_testdata/bt_goddard_ANCILLARY/'
)
