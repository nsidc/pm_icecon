from typing import Literal

Hemisphere = Literal['north', 'south']

# TODO: can we make the algorithms satellite-independent? Currently code
# branches in e.g., `compute_bt_ic` that rely on exact string matches for some
# behavior/config.
ValidSatellites = Literal['u2', '17', '18', '00', 'a2l1c']
