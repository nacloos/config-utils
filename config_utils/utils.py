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

from multisys.utils.path_utils import MultiPath


# TODO: use the type hints to indicate which arg is SavePath
def loading_function(path_arg: int = 0):
    assert isinstance(path_arg, int)

    def load_function_wrapper(func):
        # add the functionality of trying to load from multiple paths

        @wraps(func)
        def func_multiple_paths(*args, **kwargs):
            # isolate the path argument from args
            # TODO: same for kwargs if path_arg is a str and not a int
            paths = args[path_arg]
            args = list(args)

            if isinstance(paths, MultiPath):
                for p in paths:
                    if Path(p).exists():
                        args[path_arg] = p
                        return func(*args, **kwargs)

                raise ImportError(repr(paths))
            else:
                return func(*args, **kwargs)

        return func_multiple_paths
    return load_function_wrapper


def save_load(func, path, config, load=True):
    """
    Wrap the function so that the outputs are automatically saved and loaded when the function is called and the config
    is the same.
    """
    path = Path(path) if isinstance(path, str) else path
    config_hash = hash_dict(config)
    save_path = path / func.__name__ / config_hash
    data_path = save_path / "data.pkl"

    @wraps(func)
    def func_save_load(*args, **kwargs):
        if load and data_path.exists():
            # load the output
            output = load_pickle(data_path)
        else:
            # compute and save the output
            output = func(*args, **kwargs)
            save_pickle(data_path, output)
            save_config(config, save_path, "config.json")

        return output

    return func_save_load


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


def saveres(save_path, res):
    save_path = Path(save_path)
    save_path = save_path.parents[0]/save_path.stem.replace('.', '-')
    save_path.parents[0].mkdir(parents=True, exist_ok=True)
    np.save(save_path, res)


@loading_function()
def loadres(path):
    res = np.load(path, allow_pickle=True)
    if len(res.shape) == 0:
        res = res.item()
    return res


def isimportable_attr(module_and_attr_name):
    module_name, attr_name = module_and_attr_name.rsplit('.', 1)
    spec = importlib.util.find_spec(module_name)
    if spec is not None:
        # TODO: not working
        # module = importlib.util.module_from_spec(spec)
        # if hasattr(module, attr_name):
        #     return True
        # else:
        #     return False

        try:
            import_attr(module_and_attr_name)
            return True
        except AttributeError:
            return False
    else:
        return False
    # except ModuleNotFoundError as e:
    #     return False
    # except AttributeError as e:
    #     return False
    # TODO: too general to catch Exception here
    # except Exception:
    #     traceback.print_exc()
    #     return False


def import_attr(module_and_attr_name):
    """
    Import an attr (class or function) from a (python) module, e.g. 'models.RNN' (taken from the Full Stack DL course)
    Args:
        module_and_attr_name: if list of str, try to import each attr in order and return the first that can be imported
    """
    if isinstance(module_and_attr_name, list):
        attr_ = None
        # try all the paths in the list
        for path in module_and_attr_name:
            if isimportable_attr(path):
                attr_ = import_attr(path)
                break

        if attr_ is None:
            raise ValueError("Attr not found {}".format(module_and_attr_name))

    else:
        if "." in module_and_attr_name:
            module_name, class_name = module_and_attr_name.rsplit(".", 1)
        else:
            # TODO: empty module error
            module_name = "."
            class_name = module_and_attr_name
        module = importlib.import_module(module_name)
        attr_ = getattr(module, class_name)

    return attr_


def import_module_from_path(file_path):
    # https://docs.python.org/3/library/importlib.html#importing-a-source-file-directly
    file_path = Path(file_path) if isinstance(file_path, str) else file_path
    module_name = file_path.stem
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def parse_space_spec(space):
    """
    Args:
         space: input or output space specification
         e.g. [('fixation', 1), ('state', 6)] => {'fixation': [0], 'state': range(1, 7)}
    """
    if isinstance(space, dict) or space is None:
        return space
    elif isinstance(space, list):
        space_dict = {}
        idx = 0
        for (label, dim) in space:
            space_dict[label] = idx + np.arange(dim)
            idx += dim
        return space_dict
    else:
        raise TypeError("Unexpected type {}".format(space))


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
