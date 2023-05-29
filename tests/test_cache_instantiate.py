from omegaconf import OmegaConf

from config_utils.utils import get_external_dependencies
from config_utils.instantiate import instantiate, set_cache_mode


def test_cache_instantiate():
    set_cache_mode('key')
    cfg = OmegaConf.create({
        "x": {"ctx": {
            "_target_": "config_utils.utils.mock_function",
            "text": "hello"
        }},
        "instantiate": [
            "${x}",
            "${x}"
        ]
    })
    print(instantiate(cfg.instantiate))


def test_get_external_dependencies():
    cfg = OmegaConf.create({
        "param": 0,
        "f": {
            "x": "This is a test for ${.y} and ${param}",
            "y": 10
        }
    })
    dep = get_external_dependencies(cfg.f)
    assert dep == ["param"]

def test_dynamic_config():
    cfg = OmegaConf.create({
        "param": 0,
        "f": {
            "_target_": "config_utils.utils.mock_function",
            "param": "${param}",
            "_partial_": True
        }
    })
    f = instantiate(cfg.f)
    print(f())
    cfg.param = 1
    print(f())

    # this problem arises when sampling config and having a wrapper function with _partial_ = True, where the params depend on some sampled values

"""
wrappers:
  break_win:
    _inp_: ${v:rule.goal}  # whenever one of the input changes, the cached object has to be updated
    _target_: baba_is_ai.env.symbolic_env.move_obj_wrapper
    _partial_: true
    rule: ${v:rule.goal}
    block_to_move: [0, 1, 2, 3]
"""


def test_cache_sample():
    cfg = OmegaConf.create({
        "objects": ["ball", "key"],
        "env": {
            "o": "${objects}",
            "wrapper": {
                "_target_": "config_utils.utils.mock_function",
                "o": "${..o}",
                "_partial_": True
            },
            "_sample_": {
                "keys": ["o"]
            }
        }
    })
