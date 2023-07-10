from functools import cached_property

# Change in new pydantic
# from pydantic import BaseModel, Extra
from pydantic import BaseModel, ConfigDict, ValidationError


# class ConfigBaseModel(BaseModel, extra='forbid'):
class ConfigBaseModel(BaseModel):
    """Implements 'faux' immutability and allows usage of `functools.cached_property`.

    Immutability is not 'strict' (e.g., dicts can be mutated) - a
    determined dev can still mutate model instances.

    # See https://docs.pydantic.dev/dev-v2/usage/model_config/
    """

    """
    class Config:
        # Throw an error if any unexpected attrs are provided. default: 'ignore'
        # Moved to class args
        # extra = Extra.forbid

        # https://pydantic-docs.helpmanual.io/usage/models/#faux-immutability
        # Removed per new pydantic
        # allow_mutation = False

        # https://github.com/samuelcolvin/pydantic/issues/1241
        # https://github.com/samuelcolvin/pydantic/issues/2763
        # Renamed to 'ignored_types'
        # keep_untouched = (cached_property,)
        ignored_types = (cached_property,)

        arbitrary_types_allowed = True
    """
    model_config = ConfigDict(str_max_length=10)
    model_config['arbitrary_types_allowed'] = True
