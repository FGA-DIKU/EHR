from datetime import datetime
import importlib

from corebehrt.azure.util import log, check_azure, ml_client
from corebehrt.azure.util.config import (
    prepare_config,
    prepare_job_command_args,
    parse_args,
    save_config,
)


def create(
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

    return setup(
        name,
        inputs=component.INPUTS,
        outputs=component.OUTPUTS,
        config=config,
        compute=compute,
        register_output=register_output,
        log_system_metrics=log_system_metrics,
    )


def setup(
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
    save_config(job, config)

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
    from azure.ai.ml import command

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


def run(job, experiment: str):
    """
    Starts the given job in the given experiment.
    """
    check_azure()
    ml_client().create_or_update(job, experiment_name=experiment)


def run_main(
    job_name: str,
    main: callable,
    inputs: dict,
    outputs: dict,
    log_system_metrics: bool = False,
) -> None:
    """
    Implements a wrapper for running CoreBEHRT scrips on the cluster.
    Prepares input and outputs, sets up logging on Azure using MLFlow
    (if available), and finally calls the main script.

    :param job_name: Name of job
    :param main: The main callable.
    :param inputs: inputs configuration.
    :param outputs: outputs configuration.
    :param log_system_metrics: If true, logs GPU/CPU/mem usage
    """
    # Parse command line args
    args = parse_args(inputs | outputs)

    log.start_run(log_system_metrics=args.pop("log_system_metrics", False))

    cfg_path = prepare_config(job_name, args, inputs, outputs)

    main(cfg_path)

    log.end_run()
