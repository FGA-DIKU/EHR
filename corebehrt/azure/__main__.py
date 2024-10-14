import sys
import argparse

from corebehrt.common.config import load_config

from corebehrt.azure.components import create_data, create_outcomes

from . import util

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="corebehrt.azure", description="Run corebehrt jobs and pipelines in Azure"
    )

    # Global options
    parser.add_argument("-l", "--loglevel", metavar="LVL", type=str, help="Log level")
    parser.add_argument(
        "-r",
        "--rerun",
        action="store_const",
        const=True,
        default=False,
        help="Forces re-run of all jobs, even if inputs have not changed.",
    )
    parser.add_argument(
        "-e",
        "--experiment",
        type=str,
        default="corebehrt_runs",
        help="Experiment to run the job in.",
    )
    parser.add_argument(
        "-c",
        "--compute",
        type=str,
        help="Compute target to use. Default depends on job/component.",
    )
    parser.add_argument(
        "-o",
        "--register_output",
        type=str,
        action="append",
        default=[],
        help="If an output should be registered, provide a name for the Azure asset using the format '--register_output <input>=<name>'.",
    )

    # Sub-parsers
    subparsers = parser.add_subparsers(dest="call_type")
    # Job parser
    job_parser = subparsers.add_parser("job", help="Run a single job.")
    job_parser.add_argument(
        "JOB",
        type=str,
        choices=("create_data", "create_outcomes", "pretrain", "finetune"),
        help="Job to run.",
    )
    job_parser.add_argument(
        "-c",
        "--config",
        type=str,
        help="Path to configuration file. Default is file from repo.",
    )

    # Parse args
    args = parser.parse_args()

    # Parse register_output
    register_output = [o.split("=") for o in args.register_output]
    assert all(len(o) == 2 for o in register_output), "Invalid arg for register_output"
    register_output = dict(register_output)

    # Handle job
    if args.call_type == "job":

        # Path to config file
        cfg_path = (
            f"./corebehrt/configs/{args.JOB}.yaml"
            if args.config is None
            else args.config
        )
        cfg = load_config(cfg_path)

        # Setup job
        job = {
            "create_data": create_data.job,
            "create_outcomes": create_outcomes.job,
        }[
            args.JOB
        ](cfg, compute=args.compute, register_output=register_output)

        # Start job
        util.run_job(job, args.experiment)
    else:
        parser.print_help()
        sys.exit(1)
