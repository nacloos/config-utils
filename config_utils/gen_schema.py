import json
import shutil
from pathlib import Path
from subprocess import run
from typing import Callable

from config_utils.utils import json_schema_from_target
from hydra.utils import _locate as locate

JSONSCHEMA_SAVE_DIR = ".jsonschema"
CUE_SAVE_DIR = "cue_conf/schema"
SCHEMA_VERSION = "http://json-schema.org/draft/2019-09/schema#"


# TODO: {input_size: int} incompatible with {input_size: {_target_: "...", env: env}}

def remove_param_from_schema(schema, key):
    # TODO: doesn't work recursively
    if "properties" in schema:
        del schema["properties"]["self"]
    if "required" in schema:
        schema["required"] = [k for k in schema["required"] if k != "self"]


def add_param_to_schema(schema, key, typ, default, title, add_default=True, required=True):
    """
    Args:
        add_default: use an extra parameter to specify whether to add the 
            default value (cannot just use default=None as it could be a valid value)
        """
    if "properties" not in schema:
        schema["properties"] = {}
    if "required" not in schema:
        schema["required"] = []
    # TODO: not general
    schema["properties"][key] = {
        "type": typ,
        "title": title
    }
    if add_default:
        schema["properties"][key]["default"] = default

    # make the parameter required
    if required:
        schema["required"].append(key)


def generate_schemas(module_path, module, save_dir):
    """
    Args:
        module_path: module path relative to package, used to generate _target_
        module: python module where __all__ specifies which classes and functions are schematized
        save_dir: directory where the json schemas are saved
    """
    save_dir.mkdir(exist_ok=True, parents=True)

    if hasattr(module, '__all__'):
        defs = {k: v for k, v in module.__dict__.items() if k in module.__all__}
    else:
        # take all the objects in the module (except the ones starting with __ and except submodules)
        defs = {k: v for k, v in module.__dict__.items()
                if not k.startswith("__") and not isinstance(v, type(module))}

    parsed_schemas = []
    failed_schemas = []
    for k, v in defs.items():
        print("Parse to jsonschema:", k)
        if isinstance(v, type):
            target = v.__init__
            remove_self = True
        else:
            target = v
            remove_self = False

        assert isinstance(target, Callable)

        # use pydantic to generate schema from python callable
        try:
            schema = json_schema_from_target(target)
        except Exception as e:
            print(e)
            print("Error in jsonschema parsing, skip", target)
            failed_schemas.append(k)
            continue
        schema["$schema"] = SCHEMA_VERSION

        # remove self arg for class constructors
        if remove_self:
            remove_param_from_schema(schema, "self")

        # add _target_ (could also use tag @python("multisys.model.CTRNN"))
        _target_ = str(module_path).replace("/", ".") + "." + k
        # TODO: better way that modifying the json schema? (e.g. using partial or modifying the pydantic schema)
        add_param_to_schema(
            schema, key="_target_", default=_target_, typ="string",
            title="Path to locate the object (the package has to be installed)"
        )
        add_param_to_schema(schema, key="_wrap_", default=None, typ="object", title="", add_default=False, required=False)
        add_param_to_schema(schema, key="_out_", default=None, typ="object", title="", add_default=False, required=False)
        # add_param_to_schema(schema, key="_name", default=k, typ="string", title="Name of the module")

        save_file = Path(save_dir) / (k + ".json")
        with open(save_file, "w") as f:
            json.dump(schema, f, indent=4)

        parsed_schemas.append(k)

    return parsed_schemas, failed_schemas


def jsonschema_to_cue(jsonschema_dir, cue_dir, pkg_name=None):
    jsonschema_dir = Path(jsonschema_dir)
    cue_dir = Path(cue_dir)

    def run_cue_cmd(jsonschema_path: Path, cue_path: Path, pkg_name: str, def_name: str):
        cue_path.parent.mkdir(exist_ok=True, parents=True)
        # TODO: catch exception

        def_name = f"#{def_name}:"  # TODO: the # might be confusing (use close() instead?)

        out = run(["cue", "import", "-f", "-p", pkg_name, "-l", def_name, jsonschema_path])

        if out.returncode == 1:
            print("Error in cue parsing, skip", jsonschema_path)
            return False

        # move the generated cue file to the desired path
        shutil.move(jsonschema_path.with_suffix(".cue"), cue_path)
        return True

    parsed_schemas = []
    failed_schemas = []
    for p in jsonschema_dir.glob("**/*.json"):
        print("Parse to cue:", p)
        with open(p, "r") as f:
            schema = json.load(f)
        assert "$schema" in schema

        cue_path = cue_dir / Path(*p.parts[1:]).with_suffix(".cue")

        pkg_name = p.parent.name if pkg_name is None else pkg_name
        success = run_cue_cmd(jsonschema_path=p, cue_path=cue_path, pkg_name=pkg_name, def_name=p.stem)
        parsed_schemas.append(p.stem) if success else failed_schemas.append(p.stem)
    
    return parsed_schemas, failed_schemas


def generate_module_schemas(module_path, pkg_name=None, jsonschema_dir=None, cue_dir=None):
    if jsonschema_dir is None:
        jsonschema_dir = JSONSCHEMA_SAVE_DIR
    if cue_dir is None:
        cue_dir = CUE_SAVE_DIR

    jsonschema_dir = Path(jsonschema_dir) / module_path
    cue_dir = Path(cue_dir)

    module = locate(module_path.replace("/", "."))

    print("Generate schemas for:", module)
    print("-----------------------------------")

    parsed_schemas, failed_schemas = generate_schemas(module_path, module, save_dir=jsonschema_dir)
    parsed_cue_schemas, failed_cue_schemas = jsonschema_to_cue(jsonschema_dir, cue_dir, pkg_name=pkg_name)

    print("-----------------------------------")
    print("Successfully parsed to jsonschema:", parsed_schemas)
    print("Failed to parse to jsonschema:", failed_schemas)
    print("-----------------------------------")
    print("Successfully parsed to cue:", parsed_cue_schemas)
    print("Failed to parse to cue:", failed_cue_schemas)


if __name__ == '__main__':
    module_paths = [
        # "multisys/model",
        # "multiagent"
        # "netrep/metrics",
        "rsatoolbox/rdm"
    ]
    for p in module_paths:
        generate_module_schemas(p, pkg_name="rsatoolbox")
