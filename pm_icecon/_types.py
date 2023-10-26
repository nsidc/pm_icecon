from typing import Literal

# TODO: can we make the algorithms satellite-independent? Currently code
#       branches in e.g., `compute_bt_ic` that rely on exact string matches for
#       some behavior/config.
ValidSatellites = Literal[
    "u2",  # AU_SI25
    "17_class",  # f17 from CLASS (NRT)
    "17_final",  # f17 from RSS (final)
    "18_class",  # f18 data from CLASS (NRT)
    "18_final",  # f18 data from RSS (final)
    "00",  # SMMR
    "a2l1c",
]
