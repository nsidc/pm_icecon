import json
from pathlib import Path


def import_cfg_file(ifn: Path):
    return json.loads(ifn.read_text())
