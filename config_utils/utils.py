from __future__ import annotations

from urllib.request import urlopen
from io import BytesIO
from zipfile import ZipFile
import hashlib
import json
import os
import pickle
import sys
from functools import wraps
from pathlib import Path
import traceback
import importlib

import numpy as np
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf, ListConfig
from omegaconf.errors import InterpolationKeyError, ConfigKeyError, InterpolationResolutionError


def savefig(save_dir, filename, pdf=False, fig=None):
    if save_dir is None or filename is None:
        return None

    if isinstance(save_dir, str):
        save_dir = Path(save_dir)

    (Path(save_dir)/filename).parent.mkdir(parents=True, exist_ok=True)

    fig = plt.gcf() if fig is None else fig
    fig.savefig(save_dir/filename)
    fig.savefig(save_dir/(filename + ".pdf"), transparent=True) if pdf else None
    plt.close(fig)


def load_pickle(save_path):
    with open(save_path.parent / (save_path.stem + '.pkl'), 'rb') as f:
        dataset = pickle.load(f)
    return dataset


def save_pickle(save_path, dataset):
    save_path.parent.mkdir(exist_ok=True, parents=True)
    with open(save_path.parent / (save_path.stem + '.pkl'), 'wb') as f:
        pickle.dump(dataset, f)


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types
    https://stackoverflow.com/questions/26646362/numpy-array-is-not-json-serializable """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def winapi_path(dos_path, encoding=None):
    """
    https://stackoverflow.com/questions/36219317/pathname-too-long-to-open
    """
    if (not isinstance(dos_path, str) and encoding is not None):
        dos_path = dos_path.decode(encoding)
    path = os.path.abspath(dos_path)
    if path.startswith(u"\\\\"):
        return u"\\\\?\\UNC\\" + path[2:]
    return u"\\\\?\\" + path


def save_config(config, save_path: Path | str, file_name: str, to_json=False):
    if save_path is None:
        return

    save_path = Path(save_path) if isinstance(save_path, str) else save_path

    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    file_path = save_path / file_name

    # deal with long paths on windows
    if len(str(file_path.resolve())) >= 255 and os.name == 'nt':
        file_path = winapi_path(str(file_path))

    if to_json and isinstance(config, (ListConfig, DictConfig)):
        # convert to dict before saving to json
        config = OmegaConf.to_container(config)

    if isinstance(config, (ListConfig, DictConfig)):
        file_path = file_path.with_suffix('.yaml')
        OmegaConf.save(config, f=file_path)
    else:
        file_path = file_path.with_suffix('.json')
        f = open(file_path, "w")
        json.dump(config, f, indent=2, cls=NumpyEncoder)
        f.close()


def encode_dict(d):
    return json.dumps(d, sort_keys=True, cls=NumpyEncoder)


def hash_dict(d):
    # don't use python built-in hash() since not consistent across executions (random offset)
    return hashlib.sha1(bytes(encode_dict(d), 'UTF-8')).hexdigest()


def sample(dist_type: str, size: int):
    assert isinstance(dist_type, list)


def download_and_unzip(url, extract_to='.'):
    # https://gist.github.com/hantoine/c4fc70b32c2d163f604a8dc2a050d5f6
    print("Downloading: {}".format(url))
    http_response = urlopen(url)
    zipfile = ZipFile(BytesIO(http_response.read()))
    zipfile.extractall(path=extract_to)


# TODO: hydra compatibility
def call_attr(obj, attr, args=[], kwargs={}):
    return getattr(obj, attr)(*args, **kwargs)

def get_attr(obj, attr, optional=False, **kwargs):
    if optional:
        if not hasattr(obj, attr):
            return None
    return getattr(obj, attr)

def return_pi():
    import math
    return math.pi


def hash_image(img):
    import imagehash
    img_hash = imagehash.average_hash(img)
    return img_hash


def return_value(**config):
    if '_out_' in config:
        output_key = config['_out_']
    elif 'out' in config:
        output_key = config['out']
    else:
        raise ValueError
    out = config[output_key]
    return out


def return_values(config):
    return list(config.values())


def mock_function(*args, **kwargs):
    print("Call function with", args, kwargs)
    return f"Result: {args}, {kwargs}"



def get_external_dependencies(cfg):
    """
    Extract the variables that refers to values external to the config
    """
    assert isinstance(cfg, DictConfig)
    cfg = cfg.copy()
    # detach parent and try to resolve each value to identify which one depends on the parent
    cfg._set_parent(None)

    dependencies = []
    for k in cfg.keys():
        print("k", k, cfg._get_node(k))
        try:
            cfg[k]  # try to resolve the node
        except (InterpolationKeyError, InterpolationResolutionError) as e:
            # TODO: other way to extract the variable name than from the error msg?
            # error msg "Interpolation key '{var_name}' not found"
            assert len(e.msg.split("'")) == 3
            inter_key = e.msg.split("'")[1]
            dependencies.append(inter_key)
            print(e.msg)

    return dependencies

