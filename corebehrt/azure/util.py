import argparse
from os.path import join
from datetime import datetime
from typing import Tuple
import yaml
from corebehrt.azure import log
import importlib

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


def create_job(
    name: str,
    config: dict,
    compute: str,
    register_output: dict = dict(),
    log_system_metrics: bool = False,
) -> "command":  # noqa: F821
    """
    Creates the Azure command/job object. Job input/output
    configuration is loaded from the components module.
    """

    # Load component
    component = importlib.import_module(f"corebehrt.azure.components.{name}")

    return setup_job(
        name,
        inputs=component.INPUTS,
        outputs=component.OUTPUTS,
        config=config,
        compute=compute,
        register_output=register_output,
        log_system_metrics=log_system_metrics,
    )


def setup_job(
    job: str,
    inputs: dict,
    outputs: dict,
    config: dict,
    compute: str,
    register_output: dict = dict(),
    log_system_metrics: bool = False,
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
    :log_system_metrics: If true, logs GPU/CPU/mem usage
    """
    check_azure()

    # Prepare command
    cmd = f"python -m corebehrt.azure.components.{job}"

    # Make sure config is read-able -> save it in the root folder.
    with open(AZURE_CONFIG_FILE, "w") as cfg_file:
        yaml.dump(config, cfg_file)

    # Prepare input and output paths
    input_values, input_cmds = prepare_job_command_args(config, inputs, "inputs")
    output_values, output_cmds = prepare_job_command_args(
        config, outputs, "outputs", register_output=register_output
    )

    # Add input and output arguments to cmd.
    cmd += input_cmds + output_cmds

    # Add log_system_metrics if set
    if log_system_metrics:
        cmd += " --log_system_metrics"

    # Create job
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    return command(
        code=".",
        command=cmd,
        inputs=input_values,
        outputs=output_values,
        environment="CoreBEHRT@latest",
        compute=compute,
        name=f"{job}_{ts}",
    )


def run_job(job, experiment: str):
    """
    Starts the given job in the given experiment.
    """
    check_azure()
    ml_client().create_or_update(job, experiment_name=experiment)


def run_main(
    main: callable, inputs: dict, outputs: dict, log_system_metrics: bool = False
) -> None:
    """
    Implements a wrapper for running CoreBEHRT scrips on the cluster.
    Prepares input and outputs, sets up logging on Azure using MLFlow
    (if available), and finally calls the main script.

    :param main: The main callable.
    :param inputs: inputs configuration.
    :param outputs: outputs configuration.
    :param log_system_metrics: If true, logs GPU/CPU/mem usage
    """
    # Parse command line args
    args = parse_args(inputs | outputs)

    log.start_run(log_system_metrics=args.pop("log_system_metrics", False))

    prepare_config(args, inputs, outputs)

    main(AZURE_CONFIG_FILE)

    log.end_run()


def prepare_config(args: dict, inputs: dict, outputs: dict) -> None:
    """
    Prepares the config on the cluster by substituing any input/output directories
    passed as arguments in the job setup configuration file:
    -> The file (AZURE_CONFIG_FILE) is loaded.
    -> Arguments are read from the cmd-line
    -> Arguments are substituted into the configuration.
    -> The file is re-written.

    :param args: parsed arguments.
    :param inputs: input argument configuration/mapping.
    :param outputs: output argument configuration/mapping.
    """
    from corebehrt.modules.setup.config import load_config

    # Read the config file
    cfg = load_config(AZURE_CONFIG_FILE)

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
    parser.add_argument("--log_system_metrics", action="store_true", default=False)
    return vars(parser.parse_args())


def prepare_job_command_args(
    config: dict, args: dict, _type: str, register_output: dict = dict()
) -> Tuple[dict, str]:
    """
    Prepare the input/output dictionary and construct the input/output
    part of the job command string.

    :param config: Configuration dictionary.
    :param args: Job args configuration.
    :param _type: "inputs" or "outputs"
    :param register_output: Register output mapping for _type="outputs"

    :return: Tuple with input/output dictionary and argument part of command.
    """
    assert _type in ("inputs", "outputs")

    job_args = dict()
    cmd = ""
    azure_arg_cls = Input if _type == "inputs" else Output
    for arg, arg_cfg in args.items():
        if value := get_path_from_cfg(config, arg, arg_cfg):
            job_args[arg] = azure_arg_cls(path=value, type=arg_cfg["type"])

            # Update command
            cmd += " --" + arg + " ${{" + _type + "." + arg + "}}"

        # Must we register the output?
        if _type == "outputs" and arg in register_output:
            job_args[arg].name = register_output[arg]

    return job_args, cmd


def get_path_from_cfg(cfg: dict, arg: str, arg_cfg: dict):
    """
    Helper for reading a config value from config (cfg), given
    the argument name (arg) and configuration (arg_cfg)
    """
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

    # Map resulting value
    return map_azure_path(cfg)


def map_azure_path(path: str) -> str:
    """
    Maps the given path to the correct format. Expect format of input path is:

        azureml:* => Already Azure path, return as is.
        {researher-data,sp-data}:<path> => map to path on given data storage.
        {asset_name}:{asset_version} => map to Azure asset.
        * => assumed local path, return as is.
    """
    if ":" not in path or path.startswith("azureml:"):
        # Nothing to do.
        return path

    dstore, tail = path.split(":", 1)
    dstore = dstore.replace("-", "_")
    if dstore in ("researcher_data", "sp_data"):
        # Assumed to be path on datastore, format: <dstore>:<tail>
        return join("azureml://datastores", dstore, "paths", tail)
    else:
        # Assumed to be an asset, format <asset_name>:<asset_version>
        return f"azureml:{path}"
