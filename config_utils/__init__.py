from .config_utils import parse_config
from .omegaconf_resolvers import register_resolvers
from .utils import savefig
from .instantiate import instantiate
from .dict_utils import dict_set, dict_get, dict_del, dict_in

# register custom resolvers at package initialization (otherwise may have issues with multiprocessing)
register_resolvers()


def make(id, config_dir, instantiate_config=True, **kwargs):
    """
    :param id: convention is "/path/to/package/key.in.config"
    :param config_dir: directory of the configs
    :return: instantiated config if instantiate is True, otherwise config
    """
    from subprocess import run
    import json

    key = id.split("/")[-1]
    package = "/".join(id.split("/")[:-1])

    if not package.startswith("./"):
        package = "./" + package

    # convert cue package to json
    res = run(["cue", "export", package], capture_output=True, cwd=config_dir)

    if res.returncode == 1:
        err_msg = res.stderr.decode()
        raise Exception(err_msg)

    config = json.loads(res.stdout)
    config = dict_get(config, key)
    # instantiate the config at key
    if instantiate_config:
        return instantiate(config, **kwargs)
    else:
        return config


