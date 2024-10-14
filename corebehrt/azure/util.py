import argparse
import os
import yaml
from datetime import datetime
from corebehrt.common.config import Config, load_config

AZURE_CONFIG_FILE = "azure_job_config.yaml"
AZURE_AVAILABLE = False

try:
    from azure.ai.ml import MLClient, command, Input, Output
    from azure.identity import DefaultAzureCredential

    AZURE_AVAILABLE = True
except:
    pass


def is_azure_available() -> bool:
    global AZURE_AVAILABLE
    return AZURE_AVAILABLE


def check_azure() -> None:
    if not is_azure_available():
        raise Exception("Azure modules not found!")


def ml_client() -> "MLClient":
    check_azure()
    return MLClient.from_config(DefaultAzureCredential())


def setup_job(
    job: str,
    inputs: set,
    outputs: set,
    config: Config,
    compute: str = None,
    register_output: dict = dict(),
):
    check_azure()
    assert compute is not None

    # Prepare command
    cmd = f"python -m corebehrt.azure.components.{job}"

    # Make sure config is read-able -> save it in the root folder.
    config.save_to_yaml(AZURE_CONFIG_FILE)

    # Input paths
    input_values = dict()
    for arg, cfg_path in inputs.items():
        value = config
        for step in cfg_path:
            value = value[step]
        input_values[arg] = Input(path=value, type="uri_folder")

        # Update command
        cmd += " --" + arg + " ${{inputs." + arg + "}}"

    # Output paths
    output_values = dict()
    for arg, cfg_path in outputs.items():
        value = config
        for step in cfg_path:
            value = value[step]
        output_values[arg] = Output(path=value, type="uri_folder")

        # Update command
        cmd += " --" + arg + " ${{outputs." + arg + "}}"

        # Must we register the output?
        if arg in register_output:
            output_values[arg].name = register_output[arg]

    # Create job
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    return command(
        code=".",
        command=cmd,
        inputs=input_values,
        outputs=output_values,
        environment="PHAIR:23",
        compute=compute,
        name=f"{job}_{ts}",
    )


def run_job(job, experiment: str):
    check_azure()
    ml_client().create_or_update(job, experiment_name=experiment)


def prepare_config(cmd: str, inputs: set, outputs: set) -> None:
    # Read the config file
    cfg = load_config(AZURE_CONFIG_FILE)

    # Parse command line args
    args = parse_args(cmd, inputs | outputs)

    # Update input arguments in config file
    for arg, cfg_path in (inputs | outputs).items():
        assert args[arg] is not None, f"Missing argument '{arg}'"
        _cfg = cfg
        for step in cfg_path[:-1]:
            _cfg = _cfg[step]
        _cfg[cfg_path[-1]] = args[arg]

    # Overwrite config file
    cfg.save_to_yaml(AZURE_CONFIG_FILE)


def parse_args(cmd: str, args: set) -> dict:
    parser = argparse.ArgumentParser(prog=f"corebehrt.azure.{cmd}")
    for arg in args:
        parser.add_argument(f"--{arg}", type=str)
    return vars(parser.parse_args())
