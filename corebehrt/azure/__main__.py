"""
Command-line utility for starting CoreBEHRT jobs on Azure clusters.
Requires installation of azure-ml-ai python package and a valid Azure workspace.
"""

import sys
import argparse
import yaml
from . import environment, util


def parse_pair_args(pair_args: list) -> dict:
    """
    Parses the append arguments.
    Each argument is expected to be of the form <key>=<value>

    :param args: List of arguments to be parsed.

    :return: a dict/mapping from key to value
    """
    pairs = [p.split("=") for p in pair_args]
    assert all(len(p) == 2 for p in pairs), "Invalid paired arg..."
    return dict(pairs)


def create_and_run_job(args) -> None:
    """
    Run the job from the given arguments.
    """

    cfg_path = args.config or f"./corebehrt/configs/{args.JOB}.yaml"
    with open(cfg_path, "r") as cfg_file:
        cfg = yaml.safe_load(cfg_file)

    register_output = parse_pair_args(args.register_output)

    job = util.job.create(
        args.JOB,
        cfg,
        compute=args.COMPUTE,
        register_output=register_output,
        log_system_metrics=args.log_system_metrics,
    )

    # Start job
    util.job.run(job, args.experiment)


def create_and_run_pipeline(args) -> None:
    """
    Run the pipeline from the given arguments
    """

    # Read configs
    configs = dict()
    cfg_paths = parse_pair_args(args.config)
    for job_name, cfg_path in cfg_paths.items():
        with open(cfg_path, "r") as cfg_file:
            configs[job_name] = yaml.safe_load(cfg_file)

    # Parse computes and set default
    computes = parse_pair_args(args.compute)
    if args.COMPUTE is not None:
        computes["default"] = args.COMPUTE

    # Parse register_output
    register_output = parse_pair_args(args.register_output)

    pl = util.pipeline.create(
        args.PIPELINE,
        configs,
        computes,
        register_output=register_output,
        log_system_metrics=args.log_system_metrics,
    )

    util.pipeline.run(pl, args.experiment)


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
        choices={
            "create_data",
            "pretrain",
            "create_outcomes",
            "select_cohort",
            "finetune_cv",
        },
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
    job_parser.add_argument(
        "-lsm",
        "--log_system_metrics",
        action="store_true",
        default=False,
        help="If set, system metrics such as CPU, GPU and memory usage are logged in Azure.",
    )
    # Pipeline parser
    pl_parser = subparsers.add_parser("pipeline", help="Run a pipeline job.")
    pl_parser.add_argument(
        "PIPELINE", type=str, choices={"E2E"}, help="Pipeline to run."
    )
    pl_parser.add_argument(
        "COMPUTE",
        type=str,
        default=None,
        help="Default compute target to use. If not set, compute targets must be specified for all components.",
    )
    pl_parser.add_argument(
        "-cp",
        "--compute",
        type=str,
        action="append",
        default=[],
        help="Compute target to use for a specific pipeline step. Use: '-cp <job_name>=<compute>'.",
    )
    pl_parser.add_argument(
        "-c",
        "--config",
        type=str,
        action="append",
        default=[],
        help="Path to configuration file for each component. Use: '-c <job_name>=<path-to-cfg>'. If not set, default is file from repo.",
    )
    pl_parser.add_argument(
        "-e",
        "--experiment",
        type=str,
        default="corebehrt_runs",
        help="Experiment to run the job in.",
    )
    pl_parser.add_argument(
        "-o",
        "--register_output",
        type=str,
        action="append",
        default=[],
        help="If an output from any step in the pipeline should be registered, provide a name for the Azure asset using the format '--register_output <job_name>.<input>=<name>'.",
    )
    pl_parser.add_argument(
        "-lsm",
        "--log_system_metrics",
        action="store_true",
        default=False,
        help="If set, system metrics such as CPU, GPU and memory usage are logged in Azure.",
    )

    # Parse args
    args = parser.parse_args()

    if args.call_type == "build_env":
        # Build environment
        environment.build()
    elif args.call_type == "job":
        # Handle job
        create_and_run_job(args)
    elif args.call_type == "pipeline":
        # Handle pipelne
        create_and_run_pipeline(args)
    else:
        parser.print_help()
        sys.exit(1)
