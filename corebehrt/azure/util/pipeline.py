import importlib
from corebehrt.azure.util import check_azure, job
from corebehrt.azure.util.config import load_config, map_azure_path


def create_component(
    job_name: str,
    config_paths: dict,
    computes: dict,
    register_output: dict,
    log_system_metrics: bool,
    name: str = None,
) -> "command":  # noqa: F821
    check_azure()

    # Default component name is job_name
    name = name or job_name

    # Load config from path if given, otherwise load default
    config = load_config(
        path=config_paths.get(name),
        job_name=job_name,
        default_folder="corebehrt/azure/configs/pipeline",
    )

    # Set compute for this job
    compute = computes.get(name, computes["default"])

    # Apply all relevant output registrations
    register_output = {
        k[len(name) + 1 :]: v for k, v in register_output if k.startswith(name + ".")
    }

    return job.create(job_name, config, compute, register_output, log_system_metrics)


def create(
    name: str,
    config_paths: dict,
    computes: dict,
    register_output: dict,
    log_system_metrics: bool,
) -> "command":  # noqa: F821
    check_azure()

    from azure.ai.ml import Input

    # Load pipeline module
    pipeline_module = importlib.import_module(f"corebehrt.azure.pipelines.{name}")

    # Create pipeline command
    pipeline = pipeline_module.create(
        config_paths, computes, register_output, log_system_metrics
    )

    # Prepare pipeline inputs
    inputs = dict()
    for inp_key, inp_cfg in pipeline_module.INPUTS.items():
        # Pipeline inputs maps to an input in a config file for one of the components.
        job_name = inp_cfg["config"]
        # Load the component config
        config = load_config(path=config_paths.get(job_name), job_name=job_name)
        # Get the mounted path from the component.
        inp_path = map_azure_path(config["paths"][inp_key])

        inputs[inp_key] = Input(path=inp_path, type=inp_cfg.get("type"))

    return pipeline(**inputs)


def run(pipeline: "command", experiment: str) -> None:  # noqa: F821
    return job.run(pipeline, experiment)
