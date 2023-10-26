import json
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path
from subprocess import run
import requests
import os
import stat
import tarfile
import platform

from config_utils import parse_config, instantiate
from config_utils.config_utils import save_config


def init_cue():
    cue_version = "v0.6.0"
    binaries_dir = (Path(__file__).parent / "./cue_binaries").resolve()
    current_platform = platform.system().lower()
    assert current_platform in ["darwin", "linux", "windows"], f"Unsupported platform: {current_platform}"
    binary_name = "cue.exe" if current_platform == "windows" else "cue"

    if platform.machine().lower() in ["x86_64", "amd64"]:
        current_arch = "amd64"
    elif platform.machine().lower() in ["arm64", "aarch64"]:
        current_arch = "arm64"
    else:
        raise Exception(f"Unsupported machine: {platform.machine()}")

    cue_binary_path = binaries_dir / f"cue_{cue_version}_{current_platform}_{current_arch}" / binary_name
    assert cue_binary_path.exists(), "Cue binary not found. Please run `download_cue_binaries` first"

    # Check if the binary has execute permissions
    if not os.access(cue_binary_path, os.X_OK):
        # Add execute permissions
        current_permissions = os.stat(cue_binary_path).st_mode
        os.chmod(cue_binary_path, current_permissions | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return cue_binary_path


CUE_BINARY_PATH = init_cue()

def run_cue_cmd(cmd_args, **kwargs):
    res = run([CUE_BINARY_PATH] + cmd_args, **kwargs)
    return res


def run_config(path, config_dir, run_key="run", convert=None):
    # TODO: traceback for exceptions to refers to the right line in the cue file
    print("Export cue conf...")
    # res = run(["cue", "export", path], capture_output=True, cwd=config_dir)
    res = run_cue_cmd(["export", path], capture_output=True, cwd=config_dir)

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


def download_cue_binaries( output_dir, version="latest"):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    # GitHub API endpoint for release of CUE
    url = f"https://api.github.com/repos/cue-lang/cue/releases/{version}"

    # Fetch the release data
    response = requests.get(url)
    response.raise_for_status()
    release_data = response.json()

    platforms = ["darwin", "linux", "windows"]
    architectures = ["amd64", "arm64"]

    # Loop through the assets and download the desired binaries
    for asset in release_data['assets']:
        asset_name = asset['name']

        # Check if asset_name matches the desired platforms and architectures
        if not asset_name.endswith('.tar.gz') and not asset_name.endswith(".zip"):
            continue

        if not any(platform in asset_name for platform in platforms) or not any(arch in asset_name for arch in architectures):
            continue

        download_url = asset['browser_download_url']
        output_path = output_dir / asset_name

        print(f"Downloading {asset_name}...")
        with requests.get(download_url, stream=True) as r:
            r.raise_for_status()
            with open(output_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

        # Extract tar.gz files
        if asset_name.endswith('.tar.gz'):
            binary_dir = output_dir / asset_name.split('.tar.gz')[0]
            with tarfile.open(output_path, 'r:gz') as tar:
                tar.extractall(path=binary_dir)
            os.remove(output_path)  # remove the tar.gz file after extraction

        # Extract zip files
        elif asset_name.endswith('.zip'):
            binary_dir = output_dir / asset_name.split('.zip')[0]
            with zipfile.ZipFile(output_path, 'r') as zip_ref:
                zip_ref.extractall(binary_dir)
            os.remove(output_path)  # remove the zip file after extraction

        # Remove the doc folder
        shutil.rmtree(binary_dir / "doc")
