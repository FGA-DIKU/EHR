from corebehrt.azure.util import check_azure, job


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
