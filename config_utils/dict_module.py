from __future__ import annotations
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
        in_data = self.select_items(data, self.in_keys)

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
        
        if isinstance(out_data, (dict, DictConfig)):
            out_data = self.select_items(out_data, self.out_keys) 

        elif isinstance(out_data, (tuple, list, ListConfig)):
            assert isinstance(self.out_keys, (tuple, list, ListConfig)), self.out_keys
            # add out_keys to list
            out_data = {k: v for k, v in zip(self.out_keys, out_data)}
        else:
            # add out_keys to single output
            assert len(self.out_keys) == 1
            out_data = {self.out_keys[0]: out_data}
    
        return out_data

    def select_items(self, data: dict, keys: list):
        """
        Selects subset of keys from data dict.
        Args:
            data: dict
            keys: list of keys to output. If a key is a tuple or list [k_in, k_out], then the input key k_in is renamed to k_out.
        """
        if keys is None:
            return data

        assert isinstance(data, (dict, DictConfig)), data
        assert isinstance(keys, (tuple, list, ListConfig)), keys

        in_out_keys = []
        for k in keys:
            if isinstance(k, (tuple, list, ListConfig)):
                assert len(k) == 2
                k_in, k_out = k
            else:
                k_in, k_out = k, k
            assert dict_in(data, k_in), "Expected key '{}' not in input keys {}, module {}".format(k_in, list(data.keys()), self.module)
            in_out_keys.append((k_in, k_out))
            
        # use dict_get to handle nested keys (separated by dots)
        data = {k_out: dict_get(data, k_in) for k_in, k_out in in_out_keys}            
        
        # special case for single output and None k_out
        if len(in_out_keys) == 1 and in_out_keys[0][1] is None:
            # return the value directly
            data = data[None]

        return data

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