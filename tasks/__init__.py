# vim coc labels this an error
# from invoke import Collection
from invoke.collection import Collection

from . import format as _format
from . import test

ns = Collection()
ns.add_collection(Collection(_format))
ns.add_collection(Collection(test))
