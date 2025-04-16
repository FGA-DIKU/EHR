import importlib
from corebehrt.azure.util import check_azure, job
from corebehrt.azure.util.config import load_config, map_azure_path


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
    """
    Create the pipeline defined in corebehrt.azure.pipelines.{name} using the
    given computes and configurations.

    If test_cfg_file is given, each job type specified in the file is
    evaluated post-run.

    :param name: Name of module defining pipeline.
    :param data_path: Path to MEDS data asset input.
    :param computes: Dictionary mapping >component_name> => <compute>. The dict must
        contain "default" as well.
    :param config_paths: Dictionary mapping <component_name> => <path_to_config>. If
        a component is not set, the path <component_dir>/<component_name> will be
        used instead.
    :param register_output: A mapping <component_name>.<output_name> => <asset_name> for
        any outputs, which should be registered as assets.
    :param log_system_metrics: If True, metrics are logged for all components.
    :param test_cfg_file: Optional path to test configuration file - if set, job types
        configured in the file will be evaluated after they have run.

    :return: A pipeline (Azure command) to be run.
    """
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
        outputs = {
            k[len(name) + 1 :]: v
            for k, v in register_output.items()
            if k.startswith(name + ".")
        }

        return job.create(
            job_type,
            config,
            compute,
            outputs,
            log_system_metrics,
            test_cfg_file,
            as_component=True,
        )

    # Create pipeline command
    pipeline = pipeline_module.create(component_creator)

    # Prepare pipeline inputs - currently only data
    data_path = map_azure_path(data_path)
    data_input = Input(path=data_path, type="uri_folder")

    return pipeline(data=data_input)


def run(pipeline: "command", experiment: str) -> None:  # noqa: F821
    """
    Run the given pipeline command in the given experiment.
    """
    return job.run(pipeline, experiment)
