import json
from subprocess import run

from config_utils import parse_config, instantiate
from config_utils.config_utils import save_config


def run_config(path, config_dir, run_key="run", convert=None):
    # TODO: traceback for exceptions to refers to the right line in the cue file
    print("Export cue conf...")
    res = run(["cue", "export", path], capture_output=True, cwd=config_dir)

    if res.returncode == 1:
        err_msg = res.stderr.decode()
        raise Exception(err_msg)

    config = json.loads(res.stdout)
    # print(OmegaConf.to_yaml(config[run_key]))
    if "save_config" in config:

        save_config(config[run_key], config["save_config"])

    # convert to object when instantiating?
    inst_conf = {
        run_key: config[run_key],
    }

    if convert:
        inst_conf["_convert_"] = convert
        # "_convert_": "object"  # TODO: not working with baba because it resolves the conf before sampling

    print("Run...")
    res = instantiate(inst_conf)
    print(res)
