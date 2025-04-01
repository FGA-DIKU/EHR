import argparse
from os.path import join, dirname
from os import makedirs
from typing import Tuple
import yaml
import shutil

AZURE_CONFIG_FOLDER = ".azure_job_configs"


def config_path(
    cfg_name: str, is_job: bool = False, default_folder: str = "corebehrt/configs/"
) -> str:
    """
    Create the default path to the configuration file.

    :param cfg_name: name of file
    :param is_job: Boolean indicating whether the path is local (=False) or on
        the cluster (=True)
    :param default_folder: Folder to look for cfg_name (if is_job=False)

    :return: Path to config.
    """
    return (
        f"{AZURE_CONFIG_FOLDER}/{cfg_name}.yaml"
        if is_job
        else join(default_folder, f"{cfg_name}.yaml")
    )


def load_config(
    path: str = None, job_name: str = None, default_folder: str = "corebehrt/configs/"
) -> dict:
    """
    Load the local config from the given path.

    :param path: If given, load from this path.
    :param job_name: If path is not given, load from <default_folder>/<job_name>.yaml
    :param default_folder: Config folder if path is not given.

    :return: dict
    """
    path = path or config_path(cfg_name=job_name, default_folder=default_folder)
    with open(path, "r") as cfg_file:
        return yaml.safe_load(cfg_file)


def save_config(cfg_name: str, cfg: dict) -> None:
    """
    Save the prepared config before starting the job

    :param cfg_name: Name of config file.
    :param cfg: Dictionary to be saved.

    :return: Path to the saved config
    """
    # Make sure config is read-able -> save it in the root folder.
    path = config_path(cfg_name, is_job=True)
    makedirs(dirname(path), exist_ok=True)
    with open(path, "w") as cfg_file:
        yaml.dump(cfg, cfg_file)


def to_yaml_str(cfg: dict) -> None:
    """
    Convert the given config to a nicely formatted yaml string
    """
    return yaml.dump(cfg)


def load_job_config(cfg_name: str) -> "Config":  # noqa: F821
    """
    Load the config on the cluster

    :param cfg_name: Name of the configuration file to load.

    :return: CoreBEHRT config object.
    """
    from corebehrt.modules.setup.config import load_config as cb_load_config

    # Read the config file
    return cb_load_config(config_path(cfg_name, is_job=True))


def save_job_config(cfg_name: str, cfg: "Config") -> str:  # noqa: F821
    """
    Save the config on the cluster

    :param job_name: Name of job.
    :param cfg: CoreBEHRT config object to be saved.

    :return: Path to the saved config
    """
    path = config_path(cfg_name, is_job=True)
    makedirs(dirname(path), exist_ok=True)
    cfg.save_to_yaml(path)
    return path


def cleanup_configs():
    """
    Removes the temporary local config folder
    To be called after job submission.
    """
    shutil.rmtree(AZURE_CONFIG_FOLDER, ignore_errors=True)


def prepare_config(args: dict, inputs: dict, outputs: dict) -> str:
    """
    Prepares the config on the cluster by substituing any input/output directories
    passed as arguments in the job setup configuration file:
    -> The file is loaded.
    -> Arguments are read from the cmd-line
    -> Arguments are substituted into the configuration.
    -> The file is re-written.

    :param cfg_path: Path to the configuration file
    :param args: parsed arguments.
    :param inputs: input argument configuration/mapping.
    :param outputs: output argument configuration/mapping.

    :return: Path to the config on the cluster
    """
    # Read the config file
    cfg = load_job_config(args["config"])

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
            _cfg[step] = _cfg.get(step) or {}  # If it does not exists/is null
            _cfg = _cfg[step]
        _cfg[cfg_path[-1]] = args[arg]

    # Overwrite config file
    return save_job_config(args["config"], cfg)


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
    parser.add_argument("--config", type=str)
    parser.add_argument("--test", type=str)
    return vars(parser.parse_args())


def prepare_job_command_args(
    config: dict,
    args: dict,
    _type: str,
    register_output: dict = dict(),
    require_path: bool = True,
) -> Tuple[dict, str]:
    """
    Prepare the input/output dictionary and construct the input/output
    part of the job command string.

    :param config: Configuration dictionary.
    :param args: Job args configuration.
    :param _type: "inputs" or "outputs"
    :param register_output: Register output mapping for _type="outputs"
    :param require_path: If True, paths must be set for non-optional arguments,
        or an exception will be raised. <require_path> is always false for
        outputs, and is only true for inputs, if the job is setup as a
        pipeline command (in which case the pipeline construction sets the
        paths before job submission).

    :return: Tuple with input/output dictionary and argument part of command.
    """
    assert _type in ("inputs", "outputs")
    require_path = False if _type == "outputs" else require_path

    from azure.ai.ml import Input, Output

    job_args = dict()
    cmd = ""
    azure_arg_cls = Input if _type == "inputs" else Output
    for arg, arg_cfg in args.items():
        value = get_path_from_cfg(config, arg, arg_cfg)
        optional = arg_cfg.get("optional", False)

        if require_path and value is None:
            if optional:
                continue
            else:
                raise Exception(f"Missing required path: {arg}")

        # Set input/output
        job_args[arg] = azure_arg_cls(
            path=value, type=arg_cfg["type"], optional=optional
        )

        # Update command
        if optional and not require_path:
            # Optional pipeline inputs.
            cmd += " $[[--" + arg + " ${{" + _type + "." + arg + "}}]]"
        else:
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
        if cfg is None or step not in cfg:
            return None
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
