class CdrAmsr2Error(Exception):
    pass


class BootstrapAlgError(CdrAmsr2Error):
    pass


class UnexpectedSatelliteError(CdrAmsr2Error):
    pass
