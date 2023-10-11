import config_utils.instantiate
from omegaconf import OmegaConf, DictConfig
from omegaconf.errors import ConfigKeyError

INTERP_KEYWORD = "v"

def register_resolvers() -> None:
    OmegaConf.register_new_resolver(INTERP_KEYWORD, hierarchical_interpolation)
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.register_new_resolver("instantiate", config_utils.instantiate)


def hierarchical_interpolation(key, *, _parent_, _root_):
    assert key[0] != '.'  # no leading dot
    orig_key = key
    base_key = key.split(".")[0]
    node = _parent_
    if isinstance(node, DictConfig):
        node_keys = node.keys()  # don't want to call resolve here
    else:
        node_keys = {}

    # prevent infinite loop in cases like 'X': '${X}' (X refers to another X higher in the hierarchy)
    if base_key in node_keys:
        val = node._get_node(base_key)._value()
        if isinstance(val, str) and '${' + INTERP_KEYWORD + ":" + orig_key + '}' in val:
            # same name interpolation detected, pass to the next level directly
            node = node._get_parent_container()

    # TODO: temporary fix to prevent interpolation to parent node in cases like 'model': {'_base_': '${model}'}
    if '_base_' in node_keys:
        full_key = node._get_full_key('_base_')  # e.g. model._base_
        if base_key in full_key.split("."):
            node = node._get_parent_container()._get_parent_container()

    while True:
        if node is None:
            raise ConfigKeyError(f"Error resolving key '{key}'")
        # assert node is not None
        if isinstance(node, DictConfig) and base_key in node:
            break
        node = node._get_parent_container()
        # if node is None:
        #     raise ConfigKeyError(f"Error resolving key '{key}'")

    # return OmegaConf.select(node, key)  # not working, recursive call to resolve
    parent, last_key, value = node._select_impl(
        key,
        throw_on_missing=True,
        throw_on_resolution_failure=True
    )
    return value