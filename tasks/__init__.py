from invoke import Collection

from . import format as _format
from . import test

ns = Collection()
ns.add_collection(_format)
ns.add_collection(test)