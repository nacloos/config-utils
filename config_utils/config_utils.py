from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from glob import glob
from itertools import product

import hydra
from omegaconf import OmegaConf, DictConfig, ListConfig, ValueNode
from omegaconf._utils import _get_value
from omegaconf.errors import ConfigKeyError, InterpolationKeyError

from .instantiate import instantiate


CONFIG_DIR = 'multisys/conf'

"""
Return a copy of the _root_ config dict
"""
OmegaConf.register_new_resolver(
    name='_root_',
    resolver=lambda *, _root_: OmegaConf.to_container(_root_, resolve=False)
)
OmegaConf.register_new_resolver(
    name='_parent_',
    resolver=lambda *, _parent_: OmegaConf.to_container(_parent_, resolve=False)
)


# TODO: don't work because executed before parsing _sweep_
# OmegaConf.register_new_resolver(
#     name='select',
#     resolver=select_key_from_list
# )


def _glob(path):
    paths = glob.glob(path)
    # https://stackoverflow.com/questions/55357306/glob-glob-sorting-not-as-expected
    # paths = sorted(paths, key=lambda x: int(re.findall(r'\d+', x)[0]))
    # TODO
    if "iter" in Path(paths[0]).stem:
        paths = sorted(paths, key=lambda x: int(Path(x).stem.split("iter")[1]))

    return OmegaConf.create(paths)


OmegaConf.register_new_resolver(
    name='glob',
    resolver=_glob,
    use_cache=True
)


OmegaConf.register_new_resolver(
    name='getattr',
    resolver=lambda config, attr: getattr(instantiate(config), attr)
)


def allow_list_as_input(func, arg=0):
    """
    Allow func to take list as input.
    Args:
        func
        arg: if int, index in the args. If str, key in the kwargs.
    """
    def wrapper(*args, **kwargs):
        if isinstance(arg, int):
            args = list(args)
            inputs = args.pop(arg)
        elif isinstance(arg, str):
            inputs = kwargs.pop(arg)
        else:
            raise TypeError

        # TODO: inputs not nec the first arg
        if isinstance(inputs, list):
            outs = list()
            for inp in inputs:
                out = func(inp, *args, **kwargs)
                if isinstance(out, list):
                    outs.extend(out)
                else:
                    outs.append(out)
        else:
            outs = func(inputs, *args, **kwargs)
        return outs
    return wrapper


@allow_list_as_input
def _vary_sum_config(config, config_vary):
    """Vary config sequentially so total variations is sum of each hyperparameter's variations.
    Args:
        config: configuration to be varied
        config_vary: configuration specifying the variations
    Returns:
        configs: list of configurations
    """
    configs = list()
    for key, value in config_vary.items():
        for new_val in value:
            new_config = config.copy()
            # _nested_config_update(new_config, {key: new_val})
            OmegaConf.update(new_config, key, new_val)
            configs.append(new_config)
    return configs


@allow_list_as_input
def _vary_same_config(config, config_vary):
    """Vary config simultaneously so total variations is the same as each hyperparameter's variations."""
    configs = list()
    for value in zip(*config_vary.values()):
        new_config = config.copy()
        for k, v in zip(config_vary.keys(), value):
            OmegaConf.update(new_config, k, v)
            # _nested_config_update(new_config, {k: v})
        configs.append(new_config)
    return configs


@allow_list_as_input
def _vary_product_config(config, config_vary):
    """Vary config combinatorially so total variations is product of each hyperparameter's variations."""
    # config = OmegaConf.create(config)
    # config_vary = OmegaConf.create(config_vary)
    configs = list()
    # retrieve values of the params in config
    # combinatorial_params = {p: config[p] for p in config_vary_product}
    # loop through all possible combinations of combinatorial params
    for values in product(*config_vary.values()):
        new_config = config.copy()
        # values is a tuple with a value for each combinatorial param
        for k, v in zip(config_vary.keys(), values):
            # set the value of the combinatorial param
            # _nested_config_update(new_config, {k: v})
            OmegaConf.update(new_config, k, v)
        configs.append(new_config)
    return configs


def make_config_parser(parsers):
    def _parse_config(config: DictConfig):
        if OmegaConf.is_list(config) or isinstance(config, list):
            # have to be careful that list(config) or [cfg for cfg in config] will resolve each elem and don't want to do that here
            config = [_parse_config(config._get_node(i)) for i in range(len(config))]
            config = ListConfig(config)

        elif OmegaConf.is_dict(config) or isinstance(config, dict):
            for k in config.keys():
                # don't want to resolve the config here
                v = config._get_node(k) if isinstance(config, DictConfig) else config[k]
                config[k] = _parse_config(v)

            for parser in parsers:
                config = parser(config)

        return config
    return _parse_config


def wrap_parser(keyword='_wrap_'):
    def _wrap_config(base_config, wrapper_config):
        # base _target_ not necessary if the wrapper takes as input the config itself
        # assert '_target_' in base_config, f'{keyword} requires a base _target_, {base_config}'
        assert '_target_' in wrapper_config, f'{keyword} requires a wrapper _target_, {wrapper_config}'

        base_config = base_config.copy()
        del base_config[keyword]

        wrapper_config['_args_'] = [base_config]
        config = wrapper_config
        return config

    def _parser(config):
        if keyword in config:
            config = config.copy()  # important to copy the config to no modify it in place
            wrapper_config = config[keyword]

            if isinstance(wrapper_config, (dict, DictConfig)) and '_target_' not in wrapper_config:
                # list of wrappers
                wrapper_config = list(wrapper_config.values())

            if OmegaConf.is_list(wrapper_config) or isinstance(wrapper_config, list):
                for wrapper_cfg in wrapper_config:
                    if wrapper_cfg is None:
                        continue
                    config[keyword] = wrapper_cfg
                    config = _wrap_config(config, wrapper_cfg)
            else:
                config = _wrap_config(config, wrapper_config)
        return config
    return _parser


def base_parser(keyword):
    # TODO: use hydra search path!
    # TODO: don't work recursively with defaults
    def maybe_load_config(cfg_path, parent_config):
        if not isinstance(cfg_path, str):
            return cfg_path

        # base is a path to a config file relative to config_folder
        # base = base.replace('.', '/')  # TODO: problem with ../path/
        # config_path = (Path(config_dir) / config).with_suffix('.yaml')
        # return OmegaConf.load(config_path)

        config = hydra.compose(cfg_path)
        while len(config.keys()) == 1 and list(config.keys())[0] == '':
            config = config['']

        # same effect as @_here_
        for p in Path(cfg_path).parts[:-1]:
            if p == "\\" or p == "..":  # TODO: not general
                continue
            config = config[p]

        OmegaConf.set_struct(config, False)
        # print(OmegaConf.to_yaml(config))
        config._set_parent(parent_config)
        return config

    def merge_configs(base_config, config):
        # parse the base config
        # TODO: resolve only _base_ here?
        base_config = make_config_parser([_parser])(base_config)

        config = OmegaConf.merge(
            base_config,
            config
        )
        return config

    def _parser(config):
        if keyword not in config:
            return config

        base_config = config[keyword]

        if isinstance(base_config, (ListConfig, list)):
            if len(base_config) == 1:
                base_config = base_config[0]
            else:
                base_list = base_config
                # construct base_config by iterated merging
                base_config = base_list[0]
                for i in range(len(base_list)-1):
                    base_config = maybe_load_config(base_config, parent_config=config)
                    # base_config._set_parent(config)  # already done in maybe_load_config
                    cfg = base_list[i+1]
                    cfg = maybe_load_config(cfg, parent_config=config)

                    base_config = merge_configs(base_config, cfg)

        base_config = maybe_load_config(base_config, parent_config=config)
        config = merge_configs(base_config, config)
        del config[keyword]  # remove base after loading
        return config
    return _parser


def sweep_parser(keyword):
    def _parser(config):
        if keyword not in config:
            return config

        config = config.copy()
        sweep_config = config[keyword]

        # convert list sweep specification into dict
        if OmegaConf.is_list(sweep_config):
            sweep_dict = {}
            for k in sweep_config:
                sweep_dict[k] = config[k]  # not working with dot indexes
                del config[k]
                # sweep_dict[k] = dict_get(config, k)
                # dict_del(config, k)
            sweep_config = OmegaConf.create(sweep_dict)

        parent_config = config.copy()  # keep a copy of config to use as a parent in _parse_sweep_config

        def _parse_sweep_config(sweep_config):
            # TODO: might be a better way to do that
            # resolve only one level (test_nested_sweep_with_interpolation_case1 doesn't pass if resolve the entire config)
            sweep_config = OmegaConf.create({k: v for k, v in sweep_config.items()}, parent=parent_config)
            # make sure that the sweep_config is also parsed when have nested _sweep_
            sweep_config = make_config_parser([_parser])(sweep_config)

            # _sweep_: param: 10 => _sweep_: param: [10]
            for sweep_key, sweep_values in sweep_config.items():
                if sweep_key == '_type_':
                    continue

                if not OmegaConf.is_list(sweep_values) or isinstance(sweep_values, list):
                    sweep_values = [sweep_values]
                sweep_config[sweep_key] = sweep_values

            if '_flatten_' in sweep_config:
                def flatten(l: list | list[list]):
                    if not isinstance(l[0], (list, ListConfig)):
                        return l
                    return [item for sublist in l for item in sublist]

                for key in sweep_config['_flatten_']:
                    sweep_config[key] = flatten(sweep_config[key])
                del sweep_config['_flatten_']

            return sweep_config

        del config[keyword]

        # if specify _same_ directly without using _type_ in _sweep_
        if '_same_' in sweep_config:
            same_sweep_config = sweep_config['_same_']
            same_sweep_config = _parse_sweep_config(same_sweep_config)
            config = _vary_same_config(config, same_sweep_config)
            del sweep_config['_same_']

        sweep_config = _parse_sweep_config(sweep_config)
        sweep_type = sweep_config.pop('_type_', '_product_')

        # config is now a list of configs
        if sweep_type == '_product_':
            config = _vary_product_config(config, sweep_config)
        elif sweep_type == '_same_':
            config = _vary_same_config(config, sweep_config)
        elif sweep_type == '_sum_':
            config = _vary_sum_config(config, sweep_config)
        else:
            raise ValueError(sweep_type)

        config = OmegaConf.create(config, parent=parent_config)
        # for cfg in config:
        #     del cfg[keyword]

        # config = OmegaConf.create(config)
        # OmegaConf.resolve(config)
        return config
    return _parser


def parse_config(config: DictConfig, custom_parsers=None, config_dir=None) -> DictConfig:
    config_dir = CONFIG_DIR if config_dir is None else config_dir


    def double_underscore_to_base_parser():
        def _parser(config):
            keys_to_rename = []

            for k in config.keys():
                if not isinstance(k, str):
                    continue

                match = k[0] == '_' and k[-1] == '_'
                if not match:
                    continue
                base_key = k.split('_')[1]

                # TODO: str
                if not OmegaConf.is_dict(config[k]):
                    continue

                if '_base_' in config[k]:
                    continue

                try:
                    # test if base_key is in parent config by trying to resolve it
                    config[k]['_base_'] = '${' + base_key + '}'
                    config[k]['_base_']
                    keys_to_rename.append(k)
                except InterpolationKeyError:
                    del config[k]['_base_']

            # TODO: ok to replace _pca_: ... by pca: ...?
            for key in keys_to_rename:
                new_key = key.split('_')[1]  # TODO: not general, e.g. _test_case_ not working
                config[new_key] = config[key]
                del config[key]

            return config
        return _parser

    def groupby_parser(keyword):
        def _parser(config):
            if keyword in config:
                # instantiate groupby_config (e.g. to get a list of values for _sweep_)
                groupby_config = config[keyword]
                OmegaConf.resolve(groupby_config)
                groupby_config = parse_config(groupby_config)
                # print(OmegaConf.to_yaml(groupby_config))

                values = instantiate(groupby_config)
                del config[keyword]
                values = list(values)
                # print(values)
                config = OmegaConf.structured(values, flags={"allow_objects": True})


            return config
        return _parser

    config = OmegaConf.structured(config, flags={"allow_objects": True})

    # config = make_config_parser([double_underscore_to_base_parser()])(config)  # TODO: might cause some problems with resolve hierachically
    config = make_config_parser([base_parser("_base_")])(config)
    # config = make_config_parser([groupby_parser("_groupby_")])(config)
    config = make_config_parser([sweep_parser("_sweep_")])(config)

    # TODO: choose when to apply the custom parser
    if custom_parsers is not None:
        custom_parsers = [custom_parsers] if not isinstance(custom_parsers, (list, ListConfig)) else custom_parsers
        for parser in custom_parsers:
            config = parser(config)

    # _base_ and _sweep_ before resolve and _wrap_ after
    # config = OmegaConf.create(config)
    config = OmegaConf.structured(config, flags={"allow_objects": True})
    # OmegaConf.resolve(config)

    # parse _wrap_ after parsing _sweep_ since it changes param refs by nesting _target_ configs
    # config = make_config_parser([wrap_parser("_wrap_")])(config)

    return config


def save_config(config, f):
    """Save config to file."""
    OmegaConf.save(config, f)


def list_to_filename(l, separator='-'):
    if not isinstance(l, list):
        return str(l)
    name = ""
    for i, elem in enumerate(l):
        if i > 0:
            name += separator
        # yang19 tasks formatting
        if isinstance(elem, str) and 'yang19' in elem:
            elem = elem.split('.')[1].split('-')[0]
        name += str(elem)
    return name


def load_config(path, resolve=True, to_dict=True):
    cfg = hydra.compose(config_name=path)
    cfg = parse_config(cfg)

    # env/ring_manip/config => config
    for idx in path.split("/")[:-1]:
        cfg = cfg[idx]

    if to_dict:
        cfg = OmegaConf.to_container(cfg, resolve=resolve)
    return cfg
