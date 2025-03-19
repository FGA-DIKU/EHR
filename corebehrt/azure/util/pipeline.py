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
    computes: dict,
    config_paths: dict = None,
    config_dir: str = None,
    register_output: dict = None,
    log_system_metrics: bool = False,
    test_cfg_file: str = None,
) -> "command":  # noqa: F821
    check_azure()

    assert (
        config_paths is not None or config_dir is not None
    ), "Either config_paths or config_dir must be set"

    config_paths = config_paths or {}
    register_output = register_output or {}

    from azure.ai.ml import Input

    # Load pipeline module
    pipeline_module = importlib.import_module(f"corebehrt.azure.pipelines.{name}")

    # Component creator with context
    def component_creator(job_type: str, name: str = None):
        name = name or job_type

        compute = computes.get(name, computes["default"])
        # config_path = config_paths.get(name, join(config_dir, f"{name}.yaml"))
        config = load_config(
            path=config_paths.get(name), job_name=name, default_folder=config_dir
        )

        # Apply all relevant output registrations
        register_output = {
            k[len(name) + 1 :]: v
            for k, v in register_output
            if k.startswith(name + ".")
        }

        return job.create(
            job_type,
            config,
            compute,
            register_output,
            log_system_metrics,
            test_cfg_file,
        )

    # Create pipeline command
    pipeline = pipeline_module.create(component_creator)
    #    config_paths, computes, register_output, log_system_metrics, test_cfg_file
    # )

    # Prepare pipeline inputs - currently only data
    data_path = map_azure_path(data_path)
    data_input = Input(path=data_path, type="uri_folder")

    return pipeline(data=data_input)


def run(pipeline: "command", experiment: str) -> None:  # noqa: F821
    return job.run(pipeline, experiment)
