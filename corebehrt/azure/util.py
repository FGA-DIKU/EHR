import argparse
import os
import yaml

from azure.ai.ml import MLClient, command, Input, Output
from azure.identity import DefaultAzureCredential


def ml_client() -> MLClient:
    return MLClient.from_config(DefaultAzureCredential())


def flatten_definitions(definition_dct: dict):
    result = dict()
    for key, definition in definition_dct.items():
        if "type" in definition:
            result[key] = dict(definition)
        else:
            flat_sub = flatten_definitions(definition)
            for subkey, subdef in flat_sub.items():
                result[f"{key}.{subkey}"] = subdef
    return result


def unflatten(dct: dict) -> dict:
    tree = dict()
    for key, value in dct.items():
        node = tree
        path = key.split(".")
        for step in path[:-1]:
            if step not in node:
                node[step] = dict()
            node = node[step]
        node[path[-1]] = value
    return tree


def setup_component(args):
    pass


def setup_job(
    name: str,
    job: str,
    inputs: dict,
    outputs: dict,
    config: any,
    register_output: str = None,
):
    # Prepare command
    cmd = f"python -m corebehrt.azure.components.{job}"

    # Prepare inputs and outputs
    inputs = flatten_definitions(inputs)
    outputs = flatten_definitions(outputs)

    # Append to command
    cmd += "".join(" --" + a + " ${{inputs." + a + "}}" for a in inputs)

    # Set values from config or default
    def _lookup_cfg(arg, cfg, default=None):
        for step in arg.split("."):
            if step not in cfg:
                return default
            cfg = cfg[step]
        return cfg

    ## inputs
    input_values = dict()
    for arg, definition in inputs.items():
        value = _lookup_cfg(arg, config, definition.get("default"))
        if definition["type"] == "uri_folder":
            # Create Azure Input object
            value = Input(path=value, type="uri_folder")
        elif definition.get("action") == "append":
            assert type(value) is list
            for i, value_i in enumerate(value):
                arg_i = arg + "_" + str(i)
                cmd += " --" + arg_i + " ${{inputs." + arg_i + "}}"
                input_values[arg_i] = value_i
        else:
            cmd += " --" + arg + " ${{inputs." + arg + "}}"
            input_values[arg] = value
    ## Outputs
    output_values = dict()
    for arg, definition in outputs.items():
        value = _lookup_cfg(arg, config, definition.get("default"))
        if definition["type"] == "uri_folder":
            # Create Azure Input object
            value = Output(path=value, type="uri_folder")
            if register_output is not None:
                value.name = register_output
        cmd += " --" + arg + " ${{outputs." + arg + "}}"
        output_values[arg] = value

    return command(
        code=".",
        command=cmd,
        inputs=inputs,
        outputs=outputs,
        environment="PHAIR",
        compute="CPU-20-LP",
        name=name,
    )


def run_job(job, experiment: str):
    ml_client().create_or_update(job, experiment_name=experiment)


def parse_config(cmd: str, arg_cfg: dict, save_to_dir: str = None) -> "config":
    args = parse_args(cmd, arg_cfg)
    config = unflatten(vars(args))
    if save_to_dir is None:
        return config
    else:
        file_path = os.path.join(save_to_dir, f"{cmd}.yaml")
        os.makedirs(save_to_dir, exist_ok=True)
        with open(file_path, "w") as f:
            yaml.dump(config, f)
        return file_path


def parse_args(cmd: str, args: dict) -> None:
    parser = argparse.ArgumentParser(prog=f"corebehrt.azure.{cmd}")
    args = flatten_definitions(args)
    for arg, definition in args.items():
        _type = {"uri_folder": str}.get(definition["type"], definition["type"])
        parser.add_argument(
            f"--{arg}",
            type=_type,
            action=definition.get("action"),
            default=definition.get("default"),
            choices=definition.get("choices"),
        )
    return parser.parse_args()
