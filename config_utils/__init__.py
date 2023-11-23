from .config_utils import parse_config
from .omegaconf_resolvers import register_resolvers
from .utils import savefig
from .instantiate import instantiate
from .dict_utils import dict_set, dict_get, dict_del, dict_in
from config_utils.cue_utils import run_cue_cmd

# register custom resolvers at package initialization (otherwise may have issues with multiprocessing)
register_resolvers()



def make(id=None, config_dir=None, package=None, key=None, instantiate_config=True, return_config=False, 
         cached_config=None, **kwargs):
    """
    :param id: convention is "/path/to/package/key.in.config"
    :param config_dir: directory of the configs
    :return: instantiated config if instantiate is True, otherwise config
    """
    from subprocess import run
    import json

    # TODO: use two separate args instead of one arg id?
    if package is None and key is None:
        key = id.split("/")[-1]
        package = "/".join(id.split("/")[:-1])

    if not package.startswith("./"):
        package = "./" + package

    # print("Package:", package)
    # print("Key:", key)
    # convert cue package to json
    if cached_config is None:
        # export cue config
        res = run_cue_cmd(["export", package], capture_output=True, cwd=config_dir)

        if res.returncode == 1:
            err_msg = res.stderr.decode()
            raise Exception(err_msg)

        config = json.loads(res.stdout)
    else:
        # use the given cached config
        config = cached_config

    if key is not None:
        if not dict_in(config, key):
            raise Exception(f"Key {key} not found in config package {package}")
        config = dict_get(config, key)

    # instantiate the config at key
    if instantiate_config and not return_config:
        return instantiate(config, **kwargs)
    else:
        return config


