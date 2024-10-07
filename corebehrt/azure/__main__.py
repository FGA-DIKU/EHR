import sys
import argparse
from corebehrt.common.config import load_config
from corebehrt.azure import components as C

from . import util

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="corebehrt.azure", description="Run corebehrt jobs and pipelines in Azure"
    )

    # Global options
    parser.add_argument("EXPERIMENT", type=str, help="Experiment to run the job in.")
    parser.add_argument("-l", "--loglevel", metavar="LVL", type=str, help="Log level")
    parser.add_argument(
        "-r",
        "--rerun",
        action="store_const",
        const=True,
        default=False,
        help="Forces re-run of all jobs, even if inputs have not changed.",
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

    # Pipeline parser
    pl_parser = subparsers.add_parser("pipeline", help="Run a pipeline.")
    pl_parser.add_argument("PIPELINE", type=str, choices=("pretrain", "finetune"))

    # Parse args
    args = parser.parse_args()

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
        job = {"create_data": C.create_data.job}[args.JOB](cfg)

        # Start job
        util.run_job(job, args.EXPERIMENT)
    elif args.call_type == "pipeline":
        assert False
    else:
        parser.print_help()
        sys.exit(1)
