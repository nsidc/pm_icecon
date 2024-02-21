"""Gives a table of NASA Team tiepoint(s).

The structure of the table -- which are nested dictionaries -- is:
    TIEPOINTS[sat][hem][freq][tiepoint]
    where:
    sat is one of:       F08, F11, F13, F15_BRIDGE, F15,
                            F16_CLASS, F17_CLASS, F18_CLASS,
                            F17_FINAL, AMSR2
    hem is one of:       n, s
    freq is one of:      19h, 19v, 37v
    tiepoint is one of:  ow, fy, my
    eg
    TIEPOINTS[f08][n][37v][ow]

These values were primarily taken from the 'seaice_goddard.f' code
of the pmalgos package.  The relevant section of this code is pasted
at the end of this file.

After further review, most of these tiepoints were copied from the
`cdralgos` repository used to generate SMMR/SSMI/SSMIS data for CDRv4:
    https://bitbucket.org/nsidc/cdralgos/src/master/nt_cdr/timeseries_constants.h
    (retrieved 02/15/2024)
"""

from typing import TypedDict

from loguru import logger
from pm_tb_data._types import Hemisphere

from pm_icecon._types import ValidSatellites


class TiePoints(TypedDict):
    # Open water
    ow: float  # noqa
    # First-year ice
    fy: float  # noqa
    # Multi-year ice
    my: float  # noqa


# Cannot use class syntax because channel names start w/ a number. Might want to
# change this.
NasateamTiePoints = TypedDict(
    "NasateamTiePoints",
    {
        "19h": TiePoints,
        "19v": TiePoints,
        "37v": TiePoints,
    },
)


TIEPOINTS: dict[str, dict[str, NasateamTiePoints]] = {
    # Source: cdralgos
    "f08": {
        "n": {
            "19h": {"ow": 113.2, "fy": 235.5, "my": 198.5},
            "19v": {"ow": 183.4, "fy": 251.5, "my": 222.1},
            "37v": {"ow": 204.0, "fy": 242.0, "my": 184.2},
        },
        "s": {
            "19h": {"ow": 117.0, "fy": 242.6, "my": 215.7},
            "19v": {"ow": 185.3, "fy": 256.6, "my": 246.9},
            "37v": {"ow": 207.1, "fy": 248.1, "my": 212.4},
        },
    },
    # Source: cdralgos
    "f11": {
        "n": {
            "19h": {"ow": 113.6, "fy": 235.3, "my": 198.3},
            "19v": {"ow": 185.1, "fy": 251.4, "my": 222.5},
            "37v": {"ow": 204.8, "fy": 242.0, "my": 185.1},
        },
        "s": {
            "19h": {"ow": 115.7, "fy": 241.2, "my": 214.6},
            "19v": {"ow": 186.2, "fy": 255.5, "my": 246.2},
            "37v": {"ow": 207.1, "fy": 245.6, "my": 211.3},
        },
    },
    # Source: cdralgos
    "f13": {
        "n": {
            "19h": {"ow": 114.4, "fy": 235.4, "my": 198.6},
            "19v": {"ow": 185.2, "fy": 251.2, "my": 222.4},
            "37v": {"ow": 205.2, "fy": 241.1, "my": 186.2},
        },
        "s": {
            "19h": {"ow": 117.0, "fy": 241.4, "my": 214.9},
            "19v": {"ow": 186.0, "fy": 256.0, "my": 246.6},
            "37v": {"ow": 206.9, "fy": 245.6, "my": 211.1},
        },
    },
    # Source: cdralgos
    "f15_bridge": {
        "n": {
            "19h": {"ow": 114.813, "fy": 235.571, "my": 198.845},
            "19v": {"ow": 184.849, "fy": 251.047, "my": 222.160},
            "37v": {"ow": 205.029, "fy": 240.749, "my": 186.124},
        },
        "s": {
            "19h": {"ow": 117.464, "fy": 241.242, "my": 214.874},
            "19v": {"ow": 185.465, "fy": 255.325, "my": 245.944},
            "37v": {"ow": 206.577, "fy": 244.426, "my": 210.685},
        },
    },
    # Source: cdralgos
    "f15": {
        # Note: All the F15 TB values are zero :-/
        "n": {
            "19h": {"ow": 0.0, "fy": 0.0, "my": 0.0},
            "19v": {"ow": 0.0, "fy": 0.0, "my": 0.0},
            "37v": {"ow": 0.0, "fy": 0.0, "my": 0.0},
        },
        "s": {
            "19h": {"ow": 0.0, "fy": 0.0, "my": 0.0},
            "19v": {"ow": 0.0, "fy": 0.0, "my": 0.0},
            "37v": {"ow": 0.0, "fy": 0.0, "my": 0.0},
        },
    },
    # Source: cdralgos
    "f16_class": {
        # Note: f16_class, f17_class, and f18_class all use same tiepoints
        "n": {
            "19h": {"ow": 116.5, "fy": 235.4, "my": 199.0},
            "19v": {"ow": 182.2, "fy": 251.7, "my": 223.4},
            "37v": {"ow": 206.5, "fy": 242.7, "my": 188.1},
        },
        "s": {
            "19h": {"ow": 118.4, "fy": 241.1, "my": 214.8},
            "19v": {"ow": 187.7, "fy": 256.2, "my": 246.9},
            "37v": {"ow": 208.9, "fy": 246.4, "my": 212.6},
        },
    },
    # Source: cdralgos
    "f17_class": {
        # Note: f16_class, f17_class, and f18_class all use same tiepoints
        "n": {
            "19h": {"ow": 116.5, "fy": 235.4, "my": 199.0},
            "19v": {"ow": 182.2, "fy": 251.7, "my": 223.4},
            "37v": {"ow": 206.5, "fy": 242.7, "my": 188.1},
        },
        "s": {
            "19h": {"ow": 118.4, "fy": 241.1, "my": 214.8},
            "19v": {"ow": 187.7, "fy": 256.2, "my": 246.9},
            "37v": {"ow": 208.9, "fy": 246.4, "my": 212.6},
        },
    },
    # Source: cdralgos
    "f18_class": {
        # Note: f16_class, f17_class, and f18_class all use same tiepoints
        "n": {
            "19h": {"ow": 116.5, "fy": 235.4, "my": 199.0},
            "19v": {"ow": 182.2, "fy": 251.7, "my": 223.4},
            "37v": {"ow": 206.5, "fy": 242.7, "my": 188.1},
        },
        "s": {
            "19h": {"ow": 118.4, "fy": 241.1, "my": 214.8},
            "19v": {"ow": 187.7, "fy": 256.2, "my": 246.9},
            "37v": {"ow": 208.9, "fy": 246.4, "my": 212.6},
        },
    },
    # Source: cdralgos
    "f17_final": {
        "n": {
            "19h": {"ow": 113.4, "fy": 232.0, "my": 196.0},
            "19v": {"ow": 184.9, "fy": 248.4, "my": 220.7},
            "37v": {"ow": 207.1, "fy": 242.3, "my": 188.5},
        },
        "s": {
            "19h": {"ow": 113.4, "fy": 237.8, "my": 211.9},
            "19v": {"ow": 184.9, "fy": 253.1, "my": 244.0},
            "37v": {"ow": 207.1, "fy": 246.6, "my": 212.6},
        },
    },
    # Source: Derived 8/18/2022 by lin reg of F17-0001 with AMSRU
    "amsru_a2": {
        "n": {
            "19h": {"ow": 109.60, "fy": 234.73, "my": 196.75},
            "19v": {"ow": 190.55, "fy": 253.07, "my": 225.80},
            "37v": {"ow": 211.20, "fy": 244.16, "my": 193.78},
        },
        "s": {
            "19h": {"ow": 110.20, "fy": 242.83, "my": 215.22},
            "19v": {"ow": 190.79, "fy": 258.78, "my": 249.71},
            "37v": {"ow": 211.90, "fy": 249.25, "my": 217.10},
        },
    },
    # Source: cdralgos
    "n07": {
        "n": {
            "19h": {"ow": 98.5, "fy": 225.2, "my": 186.8},
            "19v": {"ow": 168.7, "fy": 242.2, "my": 210.2},
            "37v": {"ow": 199.4, "fy": 239.8, "my": 180.8},
        },
        "s": {
            "19h": {"ow": 98.5, "fy": 232.2, "my": 205.2},
            "19v": {"ow": 168.7, "fy": 247.1, "my": 237.0},
            "37v": {"ow": 199.4, "fy": 245.5, "my": 210.0},
        },
    },
}  # End of TIEPOINTS{}

"""
    # This section is a template for a future NT tiepoint set
    # Source: <How these tiepoints were calculated>
    'TBsource': {
        'n': {
            '19h': {
                'ow': ,
                'fy': ,
                'my': },
            '19v': {
                'ow': ,
                'fy': ,
                'my': },
            '37v': {
                'ow': ,
                'fy': ,
                'my': }},
        's': {
            '19h': {
                'ow': ,
                'fy': ,
                'my': },
            '19v': {
                'ow': ,
                'fy': ,
                'my': },
            '37v': {
                'ow': ,
                'fy': ,
                'my': }}},
"""


def get_tiepoints(
    *,
    satellite: ValidSatellites | str,
    hemisphere: Hemisphere,
) -> NasateamTiePoints:
    """Given a satellite and hemisphere, return pre-defined tiepoints."""
    try:
        sat = {
            # TODO: we should calculate specific tiepoints for AMSRE (`ame`)
            # instead of using the AMSR2 tiepoints.
            "ame": "amsru_a2",
            "am2": "amsru_a2",
            "u2": "amsru_a2",
            "17_final": "f17_final",
            "18_class": "f18_class",
            "18_final": "f18_class",
            "F17": "f17_final",
            "F13": "f13",
            "F11": "f11",
            "F08": "f08",
            # SMMR
            "n07": "n07",
        }[satellite]
    except KeyError:
        raise NotImplementedError(
            f"No mapping between {satellite} and tiepoints table" " currently defined."
        )

    logger.info(f"Given {satellite=}, returning tiepoints for {sat}")

    return TIEPOINTS[sat][hemisphere[0].lower()]


"""
For reference, below is a copy of the /data/ statements
in lines 290-402 of the seaice_goddard.f code in the pmalgos repo:


c
c     Initialize tiepoints, (Cavalieri, et al.)
c     These arrays are populated like this:
c     data ARRAY  / N-F08, S-F08,
c                   N-F11, S-F11,
c                   N-F13, S-F13,
c                   N-F15_BRIDGE, S-F15_BRIDGE,
c                   N-F15, S-F15,
c                   N-F17, S-F17,
c                   N-F17_FINAL, S-F17_FINAL,
c                   N-F18, S-F18,  # same as F17.
c                   N-F16, S-F16  /
      data TB19VW /
     $     183.4, 185.3,        ! N-F08, S-F08
     $     185.1, 186.2,        ! N-F11, S-F11
     $     185.2, 186.0,        ! N-F13, S-F13
     $     184.849, 185.465,    ! N-F15_BRIDGE, S-F15_BRIDGE
     $     0.0, 0.0,            ! N-F15, S-F15
     $     182.2, 187.7,        ! N-F17, S-F17
     $     184.9, 184.9,        ! N-F17_FINAL, S-F17_FINAL
     $     182.2, 187.7,        ! N-F18, S-F18
     $     182.2, 187.7         ! N-F16, S-F16
     $     /
      data TB19VF /
     $     251.5, 256.6,    ! N-F08, S-F08
     $     251.4, 255.5,    ! N-F11, S-F11
     $     251.2, 256.0,    ! N-F13, S-F13
     $     251.047, 255.325,! N-F15_BRIDGE, S-F15_BRIDGE
     $     0.0, 0.0,        ! N-F15, S-F15
     $     251.7, 256.2,    ! N-F17, S-F17
     $     248.4, 253.1,    ! N-F17_FINAL, S-F17_FINAL
     $     251.7, 256.2,    ! N-F18, S-F18
     $     251.7, 256.2     ! N-F16, S-F16
     $     /
      data TB19VM /
     $     222.1, 246.9,    ! N-F08, S-F08
     $     222.5, 246.2,    ! N-F11, S-F11
     $     222.4, 246.6,    ! N-F13, S-F13
     $     222.160, 245.944,! N-F15_BRIDGE, S-F15_BRIDGE
     $     0.0, 0.0,        ! N-F15, S-F15
     $     223.4, 246.9,    ! N-F17, S-F17
     $     220.7, 244.0,    ! N-F17_FINAL, S-F17_FINAL
     $     223.4, 246.9,    ! N-F18, S-F18
     $     223.4, 246.9     ! N-F16, S-F16
     $     /

      data TB19HW /
     $     113.2, 117.0,    ! N-F08, S-F08
     $     113.6, 115.7,    ! N-F11, S-F11
     $     114.4, 117.0,    ! N-F13, S-F13
     $     114.813, 117.464,! N-F15_BRIDGE, S-F15_BRIDGE
     $     0.0, 0.0,        ! N-F15, S-F15
     $     116.5, 118.4,    ! N-F17, S-F17
     $     113.4, 113.4,    ! N-F17_FINAL, S-F17_FINAL
     $     116.5, 118.4,    ! N-F18, S-F18
     $     116.5, 118.4     ! N-F16, S-F16
     $     /
      data TB19HF /
     $     235.5, 242.6,    ! N-F08, S-F08
     $     235.3, 241.2,    ! N-F11, S-F11
     $     235.4, 241.4,    ! N-F13, S-F13
     $     235.571, 241.242,! N-F15_BRIDGE, S-F15_BRIDGE
     $     0.0, 0.0,        ! N-F15, S-F15
     $     235.4, 241.1,    ! N-F17, S-F17
     $     232.0, 237.8,    ! N-F17_FINAL, S-F17_FINAL
     $     235.4, 241.1,    ! N-F18, S-F18
     $     235.4, 241.1     ! N-F16, S-F16
     $     /
      data TB19HM /
     $     198.5, 215.7,    ! N-F08, S-F08
     $     198.3, 214.6,    ! N-F11, S-F11
     $     198.6, 214.9,    ! N-F13, S-F13
     $     198.845, 214.874,! N-F15_BRIDGE, S-F15_BRIDGE
     $     0.0, 0.0,        ! N-F15, S-F15
     $     199.0, 214.8,    ! N-F17, S-F17
     $     196.0, 211.9,    ! N-F17_FINAL, S-F17_FINAL
     $     199.0, 214.8,    ! N-F18, S-F18
     $     199.0, 214.8     ! N-F16, S-F16
     $     /

      data TB37VW /
     $     204.0, 207.1,    ! N-F08, S-F08
     $     204.8, 207.1,    ! N-F11, S-F11
     $     205.2, 206.9,    ! N-F13, S-F13
     $     205.029, 206.577,! N-F15_BRIDGE, S-F15_BRIDGE
     $     0.0, 0.0,        ! N-F15, S-F15
     $     206.5, 208.9,    ! N-F17, S-F17
     $     207.1, 207.1,    ! N-F17_FINAL, S-F17_FINAL
     $     206.5, 208.9,    ! N-F18, S-F18
     $     206.5, 208.9     ! N-F16, S-F16
     $     /
      data TB37VF /
     $     242.0, 248.1,    ! N-F08, S-F08
     $     242.0, 245.6,    ! N-F11, S-F11
     $     241.1, 245.6,    ! N-F13, S-F13
     $     240.749, 244.426,! N-F15_BRIDGE, S-F15_BRIDGE
     $     0.0, 0.0,        ! N-F15, S-F15
     $     242.7, 246.4,    ! N-F17, S-F17
     $     242.3, 246.6,    ! N-F17_FINAL, S-F17_FINAL
     $     242.7, 246.4,    ! N-F18, S-F18
     $     242.7, 246.4     ! N-F16, S-F16
     $     /
      data TB37VM /
     $     184.2, 212.4,    ! N-F08, S-F08
     $     185.1, 211.3,    ! N-F11, S-F11
     $     186.2, 211.1,    ! N-F13, S-F13
     $     186.124, 210.685,! N-F15_BRIDGE, S-F15_BRIDGE
     $     0.0, 0.0,        ! N-F15, S-F15
     $     188.1, 212.6,    ! N-F17, S-F17
     $     188.5, 212.6,    ! N-F17_FINAL, S-F17_FINAL
     $     188.1, 212.6,    ! N-F18, S-F18
     $     188.1, 212.6     ! N-F16, S-F16
     $     /
"""
