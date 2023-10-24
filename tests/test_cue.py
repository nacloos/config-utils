from config_utils.cue_utils import run_cue_cmd

import platform
def test_cue_cmd():
    print(platform.machine())
    run_cue_cmd([""])

