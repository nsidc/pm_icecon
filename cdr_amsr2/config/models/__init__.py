from cdr_amsr2.config.models.base_model import ConfigBaseModel


class FlagValues(ConfigBaseModel):
    pole_hole: int = 2510
    land: int = 2540
    missing: int = 2550
