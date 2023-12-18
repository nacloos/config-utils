import json
from omegaconf import OmegaConf
from pathlib import Path
import libtmux
import subprocess

from .instantiate import instantiate



def instantiate_tmux(session_name=None, **config):
    if session_name is not None:
        session_name = str(session_name)

    config_path = Path(".tmux") / session_name / "config.yaml"
    config_path.parent.mkdir(exist_ok=True, parents=True)
    config_path.resolve()

    # from omegaconf import OmegaConf, DictConfig, ListConfig
    # if isinstance(config, (DictConfig, ListConfig)):
    #     config = OmegaConf.to_container(config)
    # print(type(config))
    with open(config_path, "w") as f:
    #     json.dump(config, f)
       OmegaConf.save(config, f)

    code = f"""
from omegaconf import OmegaConf
from config_utils.instantiate import instantiate

with open('{config_path}', 'r') as f:
    config = OmegaConf.load(f)

instantiate(config)
"""
    print("cmd")
    cmd = f'python -c "{code}"'
    # cmd = f'python -c "from config_utils.instantiate import instantiate"'
    # with open("test.txt", "w") as f:
    #     f.write(cmd)
    # pane.send_keys(cmd)

    server = libtmux.Server()
    print(f"Starting new session {session_name}...")
    session = server.new_session(session_name=session_name)
    window = session.attached_window
    pane = window.attached_pane
    pane.send_keys(cmd)
    
    # TODO: need to activate venv
    # out = subprocess.run(["python", "-c", code], capture_output=True, text=True)
    # print(out.stdout)
    # print(out.stderr)
    print("Launched code on tmux session.")


