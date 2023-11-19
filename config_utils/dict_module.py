import numpy as np
from time import perf_counter
from typing import Any, Callable
from omegaconf import DictConfig, ListConfig
from config_utils.dict_utils import dict_get, dict_set, dict_in


class DictModule:
    def __init__(self, module, in_keys, out_keys, call_attr=None):
        """
        Args:
            module: module to call
            in_keys: list or dict of keys to get from data
            out_keys: list of keys to set in data
            call_attr: attribute of module to call
        """
        if call_attr is None:
            assert isinstance(module, Callable), module
        else:
            assert hasattr(module, call_attr), module
            assert isinstance(getattr(module, call_attr), Callable), getattr(module, call_attr)
        
        self.module = module

        self.in_keys = in_keys
        self.out_keys = out_keys
        self.call_attr = call_attr

        self.debug = False

    def __call__(self, **data) -> Any:
        tic = perf_counter()
        # use dict_get, dict_set to handle nested keys (separated by dots)
        if isinstance(self.in_keys, (tuple, list, ListConfig)):
            for k in self.in_keys:
                assert dict_in(data, k), "Expected key '{}' not in input keys {}, module {}".format(k, list(data.keys()), self.module)
            in_data = {k: dict_get(data, k) for k in self.in_keys}

        elif isinstance(self.in_keys, (dict, DictConfig)):
            for k1, k2 in self.in_keys.items():
                assert dict_in(data, k2), "Expected key '{}' not in input keys {}, module {}".format(k2, list(data.keys()), self.module)
            in_data = {k1: dict_get(data, k2) for k1, k2 in self.in_keys.items()}

        else:
            raise ValueError("in_keys must be list or dict, got {}".format(type(self.in_keys)))

        print(f"in_data: {(perf_counter() - tic)*1000:.2f}ms") if self.debug else None
        tic = perf_counter()

        if self.call_attr is None:
            out_data = self.module(**in_data)
        else:
            out_data = getattr(self.module, self.call_attr)(**in_data)


        print(f"module(): {(perf_counter() - tic)*1000:.2f}ms") if self.debug else None
        print(self.module) if self.debug else None

        if self.out_keys is None: 
            # just return the output
            return out_data
        elif len(self.out_keys) == 0:
            return None

        if out_data is None:
            assert self.out_keys is None or len(self.out_keys) == 0, self.out_keys

        # TODO: dot notation for nested keys
        
        if isinstance(out_data, (dict, DictConfig)):
             for k in self.out_keys:
                assert k in out_data, "Expected key '{}' not in output keys {}".format(k, list(out_data.keys()))
        
        elif isinstance(out_data, (tuple, list, ListConfig)):
            # add out_keys to list
            out_data = {k: v for k, v in zip(self.out_keys, out_data)}
        else:
            # add out_keys to single output
            assert len(self.out_keys) == 1
            out_data = {self.out_keys[0]: out_data}

        assert isinstance(out_data, (dict, DictConfig)), out_data
        assert isinstance(self.out_keys, (tuple, list, ListConfig)), self.out_keys
    
        return out_data

    def __getattr__(self, name):
        """
        Makes DictModule a Wrapper
        """
        if name.startswith('_'):
            raise AttributeError("attempted to get missing private attribute '{}'".format(name))
        return getattr(self.analysis, name)

    def __repr__(self):
        return "DictModule({}, in_keys={}, out_keys={})".format(self.module, self.in_keys, self.out_keys)



class DictSequential:
    def __init__(self, *modules, return_last=False):
        self.modules = modules
        self.return_last = return_last

    def __call__(self, **data) -> Any:
        # don't include the input data in the outputs
        outputs = {}  # gather outputs from each module
        for module in self.modules:
            out = module(**data)
            if out is not None:
                # each module updates the data dict
                data.update(out)
                outputs.update(out)

        if self.return_last:
            # return the last module's output
            return out
        else:
            # return all module's outputs
            return outputs

    def __repr__(self):
        s = "DictSequential(\n"
        for module in self.modules:
            s += "  {}\n".format(module)
        s += ")"
        return s