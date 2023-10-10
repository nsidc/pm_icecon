"""Test the BT params are not changing unexpectedly.

This is a starting point for ensuring that we don't tweak paramters in defined
parameter sets that are specific to a particular combination of hemisphere,
sensor, and time.
"""
import hashlib
import json
from pathlib import Path

import pm_icecon.bt.params.amsr2 as amsr2_params
import pm_icecon.bt.params.amsr2_goddard as amsr2_goddard_params


# This class stolen from qgreenland's config export code...
class MagicJSONEncoder(json.JSONEncoder):
    """Call __json__ method of object for JSON serialization.

    Also handle Paths.
    """

    def default(self, o):
        if isinstance(o, Path):
            # Not sure why Paths don't serialize out-of-the-box!
            # https://github.com/samuelcolvin/pydantic/issues/473
            return str(o)
        if hasattr(o, '__json__') and callable(o.__json__):
            return o.__json__()
        if hasattr(o, '__dict__'):
            return json.dumps(o.__dict__, cls=MagicJSONEncoder)

        return super().default(o)


# This function stolen from qgreenland's config export code...
def _get_config_hash(cfg) -> str:
    """Return a string representing the m5 of the data in `cfg`."""
    json_str = json.dumps(
        cfg,
        cls=MagicJSONEncoder,
        indent=2,
        sort_keys=True,
    )

    hash_obj = hashlib.md5(json_str.encode('utf-8'))
    digest_str = hash_obj.hexdigest()

    return digest_str


def test_amsr2_goddard_params():
    """Assert that the original params from Goddard have not changed."""
    assert (
        _get_config_hash(amsr2_goddard_params.AMSR2_NORTH_PARAMS)
        == 'd1612e6dea7635908a610205c0c0b37a'
    )
    assert (
        _get_config_hash(amsr2_goddard_params.AMSR2_SOUTH_PARAMS)
        == 'a6fea626f1f1c51de2c74831f1d99e59'
    )


def test_amsr2_params():
    """Assert that NSIDC's update to the weather filter params have not changed."""
    assert (
        _get_config_hash(amsr2_params.AMSR2_NORTH_PARAMS)
        == '8e691fd9f83cae18f186f454897c15c5'
    )
    assert (
        _get_config_hash(amsr2_params.AMSR2_SOUTH_PARAMS)
        == 'f9bd22a2d2c48a874d89c51cb2480436'
    )