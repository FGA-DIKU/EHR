from datetime import datetime
import importlib
from uuid import uuid4

from corebehrt.azure.util import log, check_azure, ml_client
from corebehrt.azure.util.suppress import suppress_warnings
from corebehrt.azure.util.config import (
    prepare_config,
    prepare_job_command_args,
    parse_args,
    save_config,
    to_yaml_str,
    cleanup_configs,
)
from corebehrt.azure.util.test import evaluate_run


def create(
    name: str,
    config: dict,
    compute: str,
    register_output: dict = dict(),
    log_system_metrics: bool = False,
    test_cfg_file: str = None,
    as_component: bool = False,
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
        test_cfg_file=test_cfg_file,
        as_component=as_component,
    )


def setup(
    job: str,
    inputs: dict,
    outputs: dict,
    config: dict,
    compute: str,
    register_output: dict = dict(),
    log_system_metrics: bool = False,
    test_cfg_file: str = None,
    as_component: bool = False,
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
    cfg_name = f"{job}_{uuid4()}"
    save_config(cfg_name, config)

    # Prepare input and output paths
    input_values, input_cmds = prepare_job_command_args(
        config, inputs, "inputs", require_path=not as_component
    )
    output_values, output_cmds = prepare_job_command_args(
        config, outputs, "outputs", register_output=register_output, require_path=False
    )

    # Add input and output arguments to cmd.
    cmd += input_cmds + output_cmds

    # Add config name to cmd
    cmd += f" --config {cfg_name}"

    # Add log_system_metrics if set
    if log_system_metrics:
        cmd += " --log_system_metrics"

    # Add test argument if test_cfg_file is set
    if test_cfg_file:
        cmd += f" --test {test_cfg_file}"

    # Description = config as yaml in code block
    description = "```\n" + to_yaml_str(config) + "```"

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
        description=description,
        name=f"{job}_{ts}",
    )


def run(job, experiment: str):
    """
    Starts the given job in the given experiment.
    """
    check_azure()

    with suppress_warnings():
        ml_client().create_or_update(job, experiment_name=experiment)

    cleanup_configs()


def run_main(
    job_name: str,
    main: callable,
    inputs: dict,
    outputs: dict,
) -> None:
    """
    Implements a wrapper for running CoreBEHRT scrips on the cluster.
    Prepares input and outputs, sets up logging on Azure using MLFlow
    (if available), and finally calls the main script.

    :param job_name: Name of job
    :param main: The main callable.
    :param inputs: inputs configuration.
    :param outputs: outputs configuration.
    """
    # Parse command line args
    args = parse_args(inputs | outputs)
    with log.start_run(log_system_metrics=args.get("log_system_metrics", False)) as run:
        run_id = run.info.run_id
        cfg_path = prepare_config(args, inputs, outputs)
        main(cfg_path)

    # Evaluate run if test param is given
    if test_cfg_file := args.get("test", False):
        evaluate_run(run_id, job_name, test_cfg_file)
