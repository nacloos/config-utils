from .config_utils import parse_config
from .omegaconf_resolvers import register_resolvers
from .utils import savefig
from .instantiate import instantiate

# register custom resolvers at package initialization (otherwise may have issues with multiprocessing)
register_resolvers()
