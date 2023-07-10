from functools import cached_property

# For Pydantic ~1.9
from pydantic import BaseModel, Extra
# For Pydantic 2.0
# from pydantic import BaseModel, ConfigDict, ValidationError


class ConfigBaseModel(BaseModel):
    """Implements 'faux' immutability and allows usage of `functools.cached_property`.

    Immutability is not 'strict' (e.g., dicts can be mutated) - a
    determined dev can still mutate model instances.

    This version is for Pydantic ~1.9
    """

    class Config:
        # Throw an error if any unexpected attrs are provided. default: 'ignore'
        extra = Extra.forbid

        # https://pydantic-docs.helpmanual.io/usage/models/#faux-immutability
        allow_mutation = False

        # https://github.com/samuelcolvin/pydantic/issues/1241
        # https://github.com/samuelcolvin/pydantic/issues/2763
        keep_untouched = (cached_property,)

        arbitrary_types_allowed = True

'''
class ConfigBaseModel(BaseModel):
    """Implements 'faux' immutability and allows usage of `functools.cached_property`.
    This version is for Pydantic 2.0
    See https://docs.pydantic.dev/dev-v2/usage/model_config/
    """
    model_config = ConfigDict(str_max_length=10)
    model_config['arbitrary_types_allowed'] = True
'''
