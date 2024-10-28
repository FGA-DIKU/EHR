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

    # Helper for reading a config value from config (cfg), given
    # the argument name (arg) and configuration (arg_cfg)
    def _get_from_cfg(cfg, arg, arg_cfg):
        path = arg_cfg.get("key", f"paths.{arg}").split(".")
        # Traverse config path
        for step in path:
            # Check if present
            if step not in cfg:
                if arg_cfg.get("optional", False):
                    # Return None if optional
                    return None
                else:
                    # Raise error
                    raise Exception(
                        f"Missing required configuration item '{'.'.join(path)}'."
                    )
            # Next step
            cfg = cfg[step]

        # Resulting value
        value = cfg

        # Check if found and raise error if not

        return value

    # Input paths
    input_values = dict()
    for arg, arg_cfg in inputs.items():
        if value := _get_from_cfg(config, arg, arg_cfg):
            input_values[arg] = Input(path=value, type=arg_cfg["type"])

            # Update command
            cmd += " --" + arg + " ${{inputs." + arg + "}}"

    # Output paths
    output_values = dict()
    for arg, arg_cfg in outputs.items():
        if value := _get_from_cfg(config, arg, arg_cfg):
            output_values[arg] = Output(path=value, type=arg_cfg["type"])

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
    for arg, arg_cfg in (inputs | outputs).items():
        if args[arg] is None:
            if arg_cfg.get("optional", False):
                continue
            else:
                raise Exception(f"Missing argument '{arg}'")
        _cfg = cfg
        cfg_path = arg_cfg.get("key", f"paths.{arg}").split(".")
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
