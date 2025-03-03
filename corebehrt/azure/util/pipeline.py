import importlib
from corebehrt.azure.util import check_azure, job
from corebehrt.azure.util.config import map_azure_path


def create_component(
    name: str,
    configs: dict,
    computes: dict,
    register_output: dict,
    log_system_metrics: bool,
) -> "command":  # noqa: F821
    check_azure()

    config = configs.get(name, dict())
    compute = computes.get(name, computes["default"])
    register_output = {
        k[len(name) + 1 :]: v for k, v in register_output if k.startswith(name + ".")
    }

    return job.create(name, config, compute, register_output, log_system_metrics)


def create(
    name: str,
    configs: dict,
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
        configs, computes, register_output, log_system_metrics
    )

    # Prepare pipeline inputs
    inputs = dict()
    for inp_key, inp_cfg in pipeline_module.INPUTS.items():
        job_name = inp_cfg["config"]
        inp_path = map_azure_path(configs[job_name]["paths"][inp_key])

        inputs[inp_key] = Input(path=inp_path, type=inp_cfg.get("type"))

    return pipeline(**inputs)


def run(pipeline: "command", experiment: str) -> None:  # noqa: F821
    return job.run(pipeline, experiment)
