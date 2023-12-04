import collections.abc
import json
from copy import deepcopy
from pathlib import Path

import yaml
import os

from config_utils.utils import winapi_path


def dict_update(d, u):
    """
    Recursively update dict (https://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth)
    """
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d
update = dict_update  # backward compatibility


def load_dict(dict_path, d=None):
    """
    Recursively load and parse dict
    Args:
        dict_path: path to a json file
        d (optional): dict or subdict loaded from dict_path
    """
    dict_path = Path(dict_path) if isinstance(dict_path, str) else dict_path
    dict_path = dict_path.resolve()
    # deal with long paths on windows
    if u"\\\\?\\" not in str(dict_path) and len(str(dict_path)) >= 255 and os.name == 'nt':
        dict_path = Path(winapi_path(str(dict_path)))

    if d is None:
        # load json file
        with open(dict_path, 'r') as f:
            if dict_path.suffix == '.json':
                d = json.load(f)
            elif dict_path.suffix == '.yaml':
                d = yaml.safe_load(f)
            else:
                raise ValueError(dict_path)

    if not isinstance(d, dict):
        raise ValueError("A dict instance is expected")

    # parse base dict
    def parse_base_dict(d, keyword='base_dict'):
        d = deepcopy(d)
        if keyword in d:
            if isinstance(d[keyword], list):
                # recursively include the dicts in the list
                if len(d[keyword]) > 0:
                    base_dict = load_dict(dict_path.parents[0] / d[keyword].pop(0))
                    update(base_dict, d)
                    d = load_dict(dict_path, base_dict)
                else:
                    del d[keyword]

            else:
                dict_file = dict_path.parents[0] / d[keyword]
                base_dict = load_dict(dict_file)

                del d[keyword]
                # overwrite base dict
                update(base_dict, d)
                d = base_dict

        return d

    d = parse_base_dict(d, keyword='base_dict')
    d = parse_base_dict(d, keyword='base')

    # parse subdicts
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = load_dict(dict_path, d=v)
        elif isinstance(v, list):
            # parse dict elements in v
            for i in range(len(v)):
                if isinstance(v[i], dict):
                    v[i] = load_dict(dict_path, d=v[i])
            d[k] = v

    return d


def _preprocess_dict(d, idx, convert2int=False, mkidx=False):
    if isinstance(idx, str):
        idx = idx.split('.')
    else:
        assert isinstance(idx, list), f"idx must be a list or str, got {type(idx)}"

    # try to convert p to int for list indexing (e.g. "key.1" => ["key"][1])
    idx = [int(p) if convert2int and str.isdigit(p) else p for p in idx]

    # make idx if not exist
    if mkidx:
        current = d
        for p in idx:
            if p not in current:
                current[p] = {}
            current = current[p]

    return d, idx

# helpers for dot dict indexing
def dict_get(d, idx, **kwargs):
    d, idx = _preprocess_dict(d, idx, **kwargs)
    for p in idx:
        d = d[p]
    return d


def dict_set(d, idx, v, **kwargs):
    d, idx = _preprocess_dict(d, idx, **kwargs)
    for p in idx[:-1]:
        d = d[p]
    d[idx[-1]] = v


def dict_in(d, idx):
    try:
        dict_get(d, idx)
        return True
    except KeyError:
        return False


def dict_del(d, idx, **kwargs):
    d, idx = _preprocess_dict(d, idx, **kwargs)
    for p in idx[:-1]:
        d = d[p]
    del d[idx[-1]]


def compare_dict(first_dict, second_dict):
    return { k : second_dict[k] for k in set(second_dict) - set(first_dict) }
