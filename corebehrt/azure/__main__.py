"""
Command-line utility for starting CoreBEHRT jobs on Azure clusters.
Requires installation of azure-ml-ai python package and a valid Azure workspace.
"""

import sys
import argparse
from corebehrt.azure import environment, util
from corebehrt.azure.util.config import load_config


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

    cfg = load_config(path=args.config, job_name=args.JOB)

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
    cfg_paths = parse_pair_args(args.config)

    # Parse computes and set default
    computes = parse_pair_args(args.compute)
    if args.COMPUTE is not None:
        computes["default"] = args.COMPUTE

    # Parse register_output
    register_output = parse_pair_args(args.register_output)

    pl = util.pipeline.create(
        args.PIPELINE,
        args.DATA,
        computes,
        config_paths=cfg_paths,
        config_dir=args.CONFIG_DIR,
        register_output=register_output,
        log_system_metrics=args.log_system_metrics,
    )

    util.pipeline.run(pl, args.experiment)


def create_and_run_test(args) -> None:
    """
    Run a pipeline test from the given args.
    """
    name = args.NAME

    test_cfg_file = f"corebehrt/azure/configs/{name}/test.yaml"
    test_cfg = load_config(test_cfg_file)

    pl = util.pipeline.create(
        "E2E",
        test_cfg["data"],
        test_cfg.get("computes", {}),
        config_dir=f"corebehrt/azure/configs/{name}",
        register_output={},
        log_system_metrics=True,
        test_cfg_file=test_cfg_file,
    )

    util.pipeline.run(pl, "corebehrt_pipeline_tests")


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
            "prepare_training_data",
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
    pl_parser.add_argument("DATA", type=str, help="Raw input data.")
    pl_parser.add_argument(
        "COMPUTE",
        type=str,
        default=None,
        nargs="?",
        help="Default compute target to use. If not set, compute targets must be specified for all components.",
    )
    pl_parser.add_argument(
        "CONFIG_DIR",
        type=str,
        default=None,
        nargs="?",
        help="Path to folder with configs. If not set, configs must be specified for all components.",
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

    # Test parser
    test_parser = subparsers.add_parser("test", help="Run a test job.")
    test_parser.add_argument("NAME", choices={"small", "full"})

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
    elif args.call_type == "test":
        # Handle test
        create_and_run_test(args)
    else:
        parser.print_help()
        sys.exit(1)
