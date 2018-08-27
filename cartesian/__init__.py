from pkg_resources import get_distribution, DistributionNotFound

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    pass

from .sklearn_api import Symbolic
from .cgp import Primitive, Symbol, Structural, Constant, Ephemeral
