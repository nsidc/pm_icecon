from cdr_amsr2.config.models.base_model import ConfigBaseModel


class FlagValues(ConfigBaseModel):
    pole_hole: int = 251
    coast: int = 253
    land: int = 254
    missing: int = 255
