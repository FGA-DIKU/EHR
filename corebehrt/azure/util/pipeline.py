import importlib
from corebehrt.azure.util import check_azure, job
from corebehrt.azure.util.config import load_config, map_azure_path


def create_component(
    job_name: str,
    config_paths: dict,
    computes: dict,
    register_output: dict,
    log_system_metrics: bool,
    test_cfg_file: str = None,
    name: str = None,
) -> "command":  # noqa: F821
    check_azure()

    # Default component name is job_name
    name = name or job_name

    # Load config from path if given, otherwise load default
    config = load_config(
        path=config_paths.get(name),
        job_name=name,
        default_folder="corebehrt/azure/configs/pipeline",
    )

    # Set compute for this job
    compute = computes.get(name, computes["default"])

    # Apply all relevant output registrations
    register_output = {
        k[len(name) + 1 :]: v for k, v in register_output if k.startswith(name + ".")
    }

    return job.create(
        job_name, config, compute, register_output, log_system_metrics, test_cfg_file
    )


def create(
    name: str,
    data_path: str,
    config_paths: dict,
    computes: dict,
    register_output: dict,
    log_system_metrics: bool,
    test_cfg_file: str = None,
) -> "command":  # noqa: F821
    check_azure()

    from azure.ai.ml import Input

    # Load pipeline module
    pipeline_module = importlib.import_module(f"corebehrt.azure.pipelines.{name}")

    # Create pipeline command
    pipeline = pipeline_module.create(
        config_paths, computes, register_output, log_system_metrics, test_cfg_file
    )

    # Prepare pipeline inputs - currently only data
    data_path = map_azure_path(data_path)
    data_input = Input(path=data_path, type="uri_folder")

    return pipeline(data=data_input)


def run(pipeline: "command", experiment: str) -> None:  # noqa: F821
    return job.run(pipeline, experiment)
