# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import copy
import functools
import inspect
from enum import Enum
from textwrap import dedent
from typing import Any, Callable, Dict, List, Sequence, Tuple, Union

from omegaconf import OmegaConf, SCMode
from omegaconf._utils import is_structured_config

from hydra._internal.utils import _locate
from hydra.errors import InstantiationException
from hydra.types import ConvertMode, TargetConf
from omegaconf.errors import InterpolationResolutionError

from config_utils.utils import get_external_dependencies

USE_CACHE = False
CACHE_MODE = None

def set_cache_usage(use_cache):
    global USE_CACHE
    USE_CACHE = use_cache


def set_cache_mode(mode):
    assert mode in [None, 'config', 'key']
    global CACHE_MODE
    CACHE_MODE = mode


class _Keys(str, Enum):
    """Special keys in configs used by instantiate."""

    TARGET = "_target_"
    CONVERT = "_convert_"
    RECURSIVE = "_recursive_"
    ARGS = "_args_"
    PARTIAL = "_partial_"


def _is_target(x: Any) -> bool:
    if isinstance(x, dict):
        return "_target_" in x
    if OmegaConf.is_dict(x):
        return "_target_" in x
    return False


def _extract_pos_args(input_args: Any, kwargs: Any) -> Tuple[Any, Any]:
    config_args = kwargs.pop(_Keys.ARGS, ())
    output_args = config_args

    if isinstance(config_args, Sequence):
        if len(input_args) > 0:
            output_args = input_args
    else:
        raise InstantiationException(
            f"Unsupported _args_ type: '{type(config_args).__name__}'. value: '{config_args}'"
        )

    return output_args, kwargs



from memoization import cached
from memoization.caching.general.keys_order_dependent import make_key

@cached(custom_key_maker=lambda _target_, _partial_, args, kwargs, full_key, node_config: make_key(node_config, {}))
def _cached_call_target(
    _target_: Callable[..., Any],
    _partial_: bool,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    full_key: str,
    node_config: Dict
) -> Any:
    """
    Use the node config as the cache key
    """

    # print("call", _target_)
    # print("\n")
    # add config to kwargs
    if '_config_' in kwargs and kwargs['_config_'] is True:
        OmegaConf.resolve(node_config)
        kwargs['config'] = node_config
        del kwargs['_config_']
        del node_config['_config_']

    return _call_target(_target_, _partial_, args, kwargs, full_key)


def _call_target(
    _target_: Callable[..., Any],
    _partial_: bool,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    full_key: str,
) -> Any:
    """Call target (type) with args and kwargs."""
    try:
        args, kwargs = _extract_pos_args(args, kwargs)
        # detaching configs from parent.
        # At this time, everything is resolved and the parent link can cause
        # issues when serializing objects in some scenarios.
        for arg in args:
            if OmegaConf.is_config(arg):
                arg._set_parent(None)
        for v in kwargs.values():
            if OmegaConf.is_config(v):
                v._set_parent(None)
    except Exception as e:
        msg = (
            f"Error in collecting args and kwargs for '{_convert_target_to_string(_target_)}':"
            + f"\n{repr(e)}"
        )
        if full_key:
            msg += f"\nfull_key: {full_key}"

        raise InstantiationException(msg) from e

    if '_selectkwargs_' in kwargs:
        selectkwargs = kwargs.pop('_selectkwargs_')
        if selectkwargs is True:
            # TODO: not working if a decorator modifies the signature of a function
            # remove kwargs that are not in the _target_ signature (i.e. the config can contain kwargs not used to instantiate the obj)
            target_signature = inspect.signature(_target_).parameters.keys()
            kwargs = {k: v for k, v in kwargs.items() if k in target_signature}

    if _partial_:
        try:
            return functools.partial(_target_, *args, **kwargs)
        except Exception as e:
            msg = (
                f"Error in creating partial({_convert_target_to_string(_target_)}, ...) object:"
                + f"\n{repr(e)}"
            )
            if full_key:
                msg += f"\nfull_key: {full_key}"
            raise InstantiationException(msg) from e
    else:
        try:
            return _target_(*args, **kwargs)
        except Exception as e:
            msg = f"Error in call to target '{_convert_target_to_string(_target_)}':\n{repr(e)}"
            if full_key:
                msg += f"\nfull_key: {full_key}"
            raise InstantiationException(msg) from e


def _convert_target_to_string(t: Any) -> Any:
    if callable(t):
        return f"{t.__module__}.{t.__qualname__}"
    else:
        return t


def _prepare_input_dict_or_list(d: Union[Dict[Any, Any], List[Any]]) -> Any:
    res: Any
    if isinstance(d, dict):
        res = {}
        for k, v in d.items():
            if k == "_target_":
                v = _convert_target_to_string(d["_target_"])
            elif isinstance(v, (dict, list)):
                v = _prepare_input_dict_or_list(v)
            res[k] = v
    elif isinstance(d, list):
        res = []
        for v in d:
            if isinstance(v, (list, dict)):
                v = _prepare_input_dict_or_list(v)
            res.append(v)
    else:
        assert False
    return res


def _resolve_target(
    target: Union[str, type, Callable[..., Any]], full_key: str
) -> Union[type, Callable[..., Any]]:
    """Resolve target string, type or callable into type or callable."""
    if isinstance(target, str):
        try:
            target = _locate(target)
        except Exception as e:
            msg = f"Error locating target '{target}', set env var HYDRA_FULL_ERROR=1 to see chained exception."
            if full_key:
                msg += f"\nfull_key: {full_key}"
            raise InstantiationException(msg) from e
    if not callable(target):
        msg = f"Expected a callable target, got '{target}' of type '{type(target).__name__}'"
        if full_key:
            msg += f"\nfull_key: {full_key}"
        raise InstantiationException(msg)
    return target


def instantiate(config: Any, *args: Any, **kwargs: Any) -> Any:
    """
    :param config: An config object describing what to call and what params to use.
                   In addition to the parameters, the config must contain:
                   _target_ : target class or callable name (str)
                   And may contain:
                   _args_: List-like of positional arguments to pass to the target
                   _recursive_: Construct nested objects as well (bool).
                                True by default.
                                may be overridden via a _recursive_ key in
                                the kwargs
                   _convert_: Conversion strategy
                        none    : Passed objects are DictConfig and ListConfig, default
                        partial : Passed objects are converted to dict and list, with
                                  the exception of Structured Configs (and their fields).
                        object  : Passed objects are converted to dict and list.
                                  Structured Configs are converted to instances of the
                                  backing dataclass / attr class.
                        all     : Passed objects are dicts, lists and primitives without
                                  a trace of OmegaConf containers. Structured configs
                                  are converted to dicts / lists too.
                   _partial_: If True, return functools.partial wrapped method or object
                              False by default. Configure per target.
    :param args: Optional positional parameters pass-through
    :param kwargs: Optional named parameters to override
                   parameters in the config object. Parameters not present
                   in the config objects are being passed as is to the target.
                   IMPORTANT: dataclasses instances in kwargs are interpreted as config
                              and cannot be used as passthrough
    :return: if _target_ is a class name: the instantiated object
             if _target_ is a callable: the return value of the call
    """
    # Return None if config is None
    if config is None:
        return None

    # TargetConf edge case
    if isinstance(config, TargetConf) and config._target_ == "???":
        # Specific check to give a good warning about failure to annotate _target_ as a string.
        raise InstantiationException(
            dedent(
                f"""\
                Config has missing value for key `_target_`, cannot instantiate.
                Config type: {type(config).__name__}
                Check that the `_target_` key in your dataclass is properly annotated and overridden.
                A common problem is forgetting to annotate _target_ as a string : '_target_: str = ...'"""
            )
        )
        # TODO: print full key

    if isinstance(config, (dict, list)):
        config = _prepare_input_dict_or_list(config)

    kwargs = _prepare_input_dict_or_list(kwargs)

    # Structured Config always converted first to OmegaConf
    if is_structured_config(config) or isinstance(config, (dict, list)):
        config = OmegaConf.structured(config, flags={"allow_objects": True})

    if OmegaConf.is_dict(config):
        # Finalize config (convert targets to strings, merge with kwargs)
        config_copy = copy.deepcopy(config)
        config_copy._set_flag(
            flags=["allow_objects", "struct", "readonly"], values=[True, False, False]
        )
        config_copy._set_parent(config._get_parent())
        config = config_copy

        if kwargs:
            config = OmegaConf.merge(config, kwargs)

        # don't want to resolve the full config directly
        # OmegaConf.resolve(config)

        _recursive_ = config.pop(_Keys.RECURSIVE, True)
        _convert_ = config.pop(_Keys.CONVERT, ConvertMode.NONE)
        _partial_ = config.pop(_Keys.PARTIAL, False)

        return instantiate_node(
            config, *args, recursive=_recursive_, convert=_convert_, partial=_partial_
        )
    elif OmegaConf.is_list(config):
        # Finalize config (convert targets to strings, merge with kwargs)
        config_copy = copy.deepcopy(config)
        config_copy._set_flag(
            flags=["allow_objects", "struct", "readonly"], values=[True, False, False]
        )
        config_copy._set_parent(config._get_parent())
        config = config_copy

        # don't want to resolve the full config directly
        # OmegaConf.resolve(config)

        _recursive_ = kwargs.pop(_Keys.RECURSIVE, True)
        _convert_ = kwargs.pop(_Keys.CONVERT, ConvertMode.NONE)
        _partial_ = kwargs.pop(_Keys.PARTIAL, False)

        if _partial_:
            raise InstantiationException(
                "The _partial_ keyword is not compatible with top-level list instantiation"
            )

        return instantiate_node(
            config, *args, recursive=_recursive_, convert=_convert_, partial=_partial_
        )
    else:
        raise InstantiationException(
            dedent(
                f"""\
                Cannot instantiate config of type {type(config).__name__}.
                Top level config must be an OmegaConf DictConfig/ListConfig object,
                a plain dict/list, or a Structured Config class or instance."""
            )
        )


def _convert_node(node: Any, convert: Union[ConvertMode, str]) -> Any:
    if OmegaConf.is_config(node):
        if convert == ConvertMode.ALL:
            node = OmegaConf.to_container(node, resolve=True)
        elif convert == ConvertMode.PARTIAL:
            node = OmegaConf.to_container(
                node, resolve=True, structured_config_mode=SCMode.DICT_CONFIG
            )
        elif convert == ConvertMode.OBJECT:
            node = OmegaConf.to_container(
                node, resolve=True, structured_config_mode=SCMode.INSTANTIATE
            )
    return node


def parse_node(node):
    from config_utils.config_utils import wrap_parser, base_parser, sweep_parser, make_config_parser

    node_keys = node.keys()  # if '_wrap_' in node => infinite loop

    # parse _base_ recursively
    node = make_config_parser([base_parser("_base_")])(node)
    # if '_base_' in node_keys:
    #     node = base_parser("_base_")(node)

    if '_sweep_' in node_keys:
        node = sweep_parser("_sweep_")(node)

    if '_wrap_' in node_keys:
        node = wrap_parser('_wrap_')(node)

    return node


def instantiate_node(
    node: Any,
    *args: Any,
    convert: Union[str, ConvertMode] = ConvertMode.NONE,
    recursive: bool = True,
    partial: bool = False,
) -> Any:
    # Return None if config is None
    if node is None or (OmegaConf.is_config(node) and node._is_none()):
        return None

    if not OmegaConf.is_config(node):
        return node

    # handle _wrap_ just before instantiating the node
    if OmegaConf.is_dict(node):
        node = parse_node(node)
        # # parse _base_ recursively
        # from config_utils.config_utils import base_parser, make_config_parser
        # node = make_config_parser([base_parser("_base_")])(node)

        # if '_sweep_' in node_keys:
        #     from config_utils.config_utils import sweep_parser
        #     node = sweep_parser("_sweep_")(node)
        #
        # # TODO: cleaner way to do that
        # if '_wrap_' in node_keys:
        #     from config_utils.config_utils import wrap_parser
        #     node = wrap_parser('_wrap_')(node)


        # if '_sample_' in node_keys:
        #     from baba_is_ai.utils.config_parser import sample_parser
        #     print("Parse sample node", node)
        #     node = sample_parser(node)
        #     # node is a generator object, don't instantiate it
        #     return node
        # node = OmegaConf.structured(node, parent=node._get_parent(), flags={"allow_objects": True})

    # Override parent modes from config if specified
    if OmegaConf.is_dict(node):
        # using getitem instead of get(key, default) because OmegaConf will raise an exception
        # if the key type is incompatible on get.
        convert = node[_Keys.CONVERT] if _Keys.CONVERT in node else convert
        recursive = node[_Keys.RECURSIVE] if _Keys.RECURSIVE in node else recursive
        partial = node[_Keys.PARTIAL] if _Keys.PARTIAL in node else partial

    full_key = node._get_full_key(None)

    if not isinstance(recursive, bool):
        msg = f"Instantiation: _recursive_ flag must be a bool, got {type(recursive)}"
        if full_key:
            msg += f"\nfull_key: {full_key}"
        raise TypeError(msg)

    if not isinstance(partial, bool):
        msg = f"Instantiation: _partial_ flag must be a bool, got {type( partial )}"
        if node and full_key:
            msg += f"\nfull_key: {full_key}"
        raise TypeError(msg)

    # If OmegaConf list, create new list of instances if recursive
    if OmegaConf.is_list(node):
        items = [
            instantiate_node(item, convert=convert, recursive=recursive)
            for item in node._iter_ex(resolve=True)
        ]

        if convert in (ConvertMode.ALL, ConvertMode.PARTIAL, ConvertMode.OBJECT):
            # If ALL or PARTIAL or OBJECT, use plain list as container
            return items
        else:
            # Otherwise, use ListConfig as container
            lst = OmegaConf.create(items, flags={"allow_objects": True})
            lst._set_parent(node)
            return lst

    elif OmegaConf.is_dict(node):
        exclude_keys = set({"_target_", "_convert_", "_recursive_", "_partial_"})
        if _is_target(node):
            _target_ = _resolve_target(node.get(_Keys.TARGET), full_key)
            kwargs = {}
            is_partial = node.get("_partial_", False) or partial

            # resolve one layer
            for key in node.keys():
                if key not in exclude_keys:
                    if OmegaConf.is_missing(node, key) and is_partial:
                        continue
                    # print(key)
                    # print(OmegaConf.to_yaml(node))
                    # print("---------")
                    value = node[key]
                    if recursive:
                        value = instantiate_node(
                            value, convert=convert, recursive=recursive
                        )
                    kwargs[key] = _convert_node(value, convert)

            if '_config_' in kwargs and kwargs['_config_'] is True:
                OmegaConf.resolve(node)
                kwargs['config'] = node
                del kwargs['_config_']
                del node['_config_']

            # TODO: deprecated
            kwargs.pop('_cache_', None)

            if CACHE_MODE is None:
                use_cache = False

            elif CACHE_MODE == 'config':
                # cannot resolve the entire config here (e.g. doesn't work with _sample_)
                if '_inp_' not in node:
                    # compute external dependencies and store them in _inp_
                    print("Compute ext dep")
                    dep = get_external_dependencies(node)
                    node['_inp_'] = OmegaConf.create(["${v:"+var+"}" for var in dep])
                print(node["_inp_"])
                # resolve only external dependencies

                try:
                    temp_node = node.copy()
                    OmegaConf.resolve(temp_node._inp_)
                    print("Caching using config:", node)
                    # print("kwargs", kwargs)
                    cache_key = temp_node
                    use_cache = True
                except InterpolationResolutionError:
                    print("Can't resolve ext dep, don't cache")
                    use_cache = False

            elif CACHE_MODE == 'key':
                # print("Caching using key:", full_key)
                cache_key = full_key
                use_cache = True
            else:
                raise ValueError

            # _inp_ is used for resolving external dependencies, not used by target
            if '_inp_' in kwargs:
                del kwargs['_inp_']


            # TODO: don't support caching partial
            if is_partial:
                use_cache = False

            if not use_cache:
                # no caching
                res = _call_target(_target_, partial, args, kwargs, full_key)
            else:
                res = _cached_call_target(_target_, partial, args, kwargs, full_key, cache_key)


            return res

            # if '_cache_' in node:
            #     _cache_ = node['_cache_']
            #     del kwargs['_cache_']
            # else:
            #     _cache_ = True  # use cache by default

            # print(full_key)

            # if USE_CACHE is False:
            #     res = _call_target(_target_, partial, args, kwargs, full_key)
            #     return res
            #
            # if not _cache_:
            #     # call target without caching
            #     res = _call_target(_target_, partial, args, kwargs, full_key)
            # else:
            #     res = _cached_call_target(_target_, partial, args, kwargs, full_key, node)
            # return res

        else:
            # If ALL or PARTIAL non structured or OBJECT non structured,
            # instantiate in dict and resolve interpolations eagerly.
            if convert == ConvertMode.ALL or (
                convert in (ConvertMode.PARTIAL, ConvertMode.OBJECT)
                and node._metadata.object_type in (None, dict)
            ):
                dict_items = {}
                for key, value in node.items():
                    # list items inherits recursive flag from the containing dict.
                    dict_items[key] = instantiate_node(
                        value, convert=convert, recursive=recursive
                    )
                return dict_items
            else:
                # Otherwise use DictConfig and resolve interpolations lazily.
                cfg = OmegaConf.create({}, flags={"allow_objects": True})
                for key, value in node.items():
                    cfg[key] = instantiate_node(
                        value, convert=convert, recursive=recursive
                    )
                cfg._set_parent(node)
                cfg._metadata.object_type = node._metadata.object_type
                if convert == ConvertMode.OBJECT:
                    return OmegaConf.to_object(cfg)
                return cfg

    else:
        assert False, f"Unexpected config type : {type(node).__name__}"
