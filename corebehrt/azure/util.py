import argparse
import os
import yaml
from datetime import datetime
from corebehrt.common.config import Config, load_config

AZURE_CONFIG_FILE = "azure_job_config.yaml"
AZURE_AVAILABLE = False

try:
    #
    # Check if azure is available and set flag.
    #
    from azure.ai.ml import MLClient, command, Input, Output
    from azure.identity import DefaultAzureCredential

    AZURE_AVAILABLE = True
except:
    pass


def is_azure_available() -> bool:
    """
    Checks if Azure modules are available.

    :return: True if available, otherwise False.
    """
    global AZURE_AVAILABLE
    return AZURE_AVAILABLE


def check_azure() -> None:
    """
    Checks if Azure modules are available, raises an exception if not.
    """
    if not is_azure_available():
        raise Exception("Azure modules not found!")


def ml_client() -> "MLClient":
    """
    Returns the Azure MLClient.
    """
    check_azure()
    return MLClient.from_config(DefaultAzureCredential())


def setup_job(
    job: str,
    inputs: dict,
    outputs: dict,
    config: Config,
    compute: str = None,
    register_output: dict = dict(),
):
    """
    Sets up the Azure job.

    :param job: Name of job.
    :param inputs: argument configuration for input directories, i.e. a mapping
        to configuration elements, typically in the paths sub-config.
    :param outputs: argument configuration for output directories.
    :param config: The config for the current job.
    :param compute: The azure compute to use for the job.
    :register_output: A mapping from output id to name, if the output should be
        registered as a data asset.
    """
    check_azure()
    assert compute is not None

    # Prepare command
    cmd = f"python -m corebehrt.azure.components.{job}"

    # Make sure config is read-able -> save it in the root folder.
    config.save_to_yaml(AZURE_CONFIG_FILE)

    # Helper for reading a config value from config (cfg), given
    # the argument name (arg) and configuration (arg_cfg)
    def _get_path_from_cfg(cfg, arg, arg_cfg):
        steps = arg_cfg.get("key", f"paths.{arg}").split(".")
        # Traverse config path
        for step in steps:
            # Check if present
            if step not in cfg:
                if arg_cfg.get("optional", False):
                    # Return None if optional
                    return None
                else:
                    # Raise error
                    raise Exception(
                        f"Missing required configuration item '{'.'.join(steps)}'."
                    )
            # Next step
            cfg = cfg[step]

        # Resulting value
        path = cfg

        # Do path mapping. Presence of ":" indicates Azure path mapping.
        if ":" in path:
            dstore, tail = path.split(":", 1)
            dstore = dstore.replace("-", "_")
            if dstore in ("researcher_data", "sp_data"):
                # Assumed to be path on datastore, format: <dstore>:<path>
                path = join("//datastores", dstore, "paths", value)

            # else: Assumed to be an asset, format <asset_name>:<asset_version>

            path = f"azureml:{path}"

        return value

    # Input paths
    input_values = dict()
    for arg, arg_cfg in inputs.items():
        if value := _get_path_from_cfg(config, arg, arg_cfg):
            input_values[arg] = Input(path=value, type=arg_cfg["type"])

            # Update command
            cmd += " --" + arg + " ${{inputs." + arg + "}}"

    # Output paths
    output_values = dict()
    for arg, arg_cfg in outputs.items():
        if value := _get_path_from_cfg(config, arg, arg_cfg):
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


def run_job(job: "Job", experiment: str):
    """
    Starts the given job in the given experiment.
    """
    check_azure()
    ml_client().create_or_update(job, experiment_name=experiment)


def prepare_config(inputs: dict, outputs: dict) -> None:
    """
    Prepares the config on the cluster by substituing any input/output directories
    passed as arguments in the job setup configuration file:
    -> The file (AZURE_CONFIG_FILE) is loaded.
    -> Arguments are read from the cmd-line
    -> Arguments are substituted into the configuration.
    -> The file is re-written.

    :param inputs: input argument configuration/mapping.
    :param outputs: output argument configuration/mapping.
    """
    # Read the config file
    cfg = load_config(AZURE_CONFIG_FILE)

    # Parse command line args
    args = parse_args(inputs | outputs)

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


def parse_args(args: set) -> dict:
    """
    Parses the arguments from the command line

    :param parse_args: The argument configuration mapping for inputs and outputs.

    :return: A dictionary mapping keys from args to the values passed to the
        command line.
    """
    parser = argparse.ArgumentParser()
    for arg in args:
        parser.add_argument(f"--{arg}", type=str)
    return vars(parser.parse_args())
