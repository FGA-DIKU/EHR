"""
Command-line utility for starting CoreBEHRT jobs on Azure clusters.
Requires installation of azure-ml-ai python package and a valid Azure workspace.
"""

import sys
import argparse

from corebehrt.modules.setup.config import load_config

from corebehrt.azure.components import (
    create_data,
    create_outcomes,
    pretrain,
    finetune,
    select_cohort,
    prepare_training_data
)

from . import environment, util

COMPONENTS = {
    "create_data": create_data,
    "create_outcomes": create_outcomes,
    "pretrain": pretrain,
    "finetune_cv": finetune,
    "select_cohort": select_cohort,
    "prepare_training_data": prepare_training_data,
}


def parse_register_output(register_output_args: list) -> dict:
    """
    Parses the register output append argument.
    Each argument is expected to be of the form <output_id>=<asset_name>

    :param args: List of arguments to be parsed.

    :return: a dict/mapping from output_id to asset_name
    """
    register_output = [o.split("=") for o in register_output_args]
    assert all(len(o) == 2 for o in register_output), "Invalid arg for register_output"
    return dict(register_output)


def get_job_initializer(name: str) -> callable:
    """
    Returns the initializer for the Azure job for the given
    component name.
    """
    return COMPONENTS[name].job


def create_and_run_job(args) -> None:
    """
    Run the job from the given arguments.
    """
    cfg = load_config(args.config or f"./corebehrt/configs/{args.JOB}.yaml")
    register_output = parse_register_output(args.register_output)
    job_initializer = get_job_initializer(args.JOB)

    job = job_initializer(cfg, compute=args.COMPUTE, register_output=register_output)

    # Start job
    util.run_job(job, args.experiment)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="corebehrt.azure", description="Run corebehrt jobs and pipelines in Azure"
    )

    # Sub-parsers
    subparsers = parser.add_subparsers(dest="call_type")
    # Environment parser
    env_parser = subparsers.add_parser(
        "build_env", help="Build the CoreBEHRT environment"
    )
    # Job parser
    job_parser = subparsers.add_parser("job", help="Run a single job.")
    job_parser.add_argument(
        "JOB",
        type=str,
        choices=COMPONENTS.keys(),
        help="Job to run.",
    )
    job_parser.add_argument(
        "COMPUTE",
        type=str,
        help="Compute target to use.",
    )
    job_parser.add_argument(
        "-c",
        "--config",
        type=str,
        help="Path to configuration file. Default is file from repo.",
    )
    job_parser.add_argument(
        "-e",
        "--experiment",
        type=str,
        default="corebehrt_runs",
        help="Experiment to run the job in.",
    )
    job_parser.add_argument(
        "-o",
        "--register_output",
        type=str,
        action="append",
        default=[],
        help="If an output should be registered, provide a name for the Azure asset using the format '--register_output <input>=<name>'.",
    )

    # Parse args
    args = parser.parse_args()

    if args.call_type == "build_env":
        # Build environment
        environment.build()
    elif args.call_type == "job":
        # Handle job
        create_and_run_job(args)
    else:
        parser.print_help()
        sys.exit(1)
