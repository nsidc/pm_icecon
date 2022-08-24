from typing import Literal

Hemisphere = Literal['north', 'south']

# TODO: can we make the algorithms satellite-independent? Currently code
#       branches in e.g., `compute_bt_ic` that rely on exact string matches for
#       some behavior/config.
ValidSatellites = Literal[
    'u2',  # AU_SI25

    # TODO: what does '17' really mean? Bootstrap code appears to use NRT/CLASS
    # data parameters but NT currently assumes 17 == RSS/final data. Maybe we
    # need to have two '17' sats here.
    '17',  # f17 data (from CLASS for bootstrap, otherwise from RSS?)

    '18',  # f18 data from CLASS
    '00',  # SMMR
    'a2l1c',
]
