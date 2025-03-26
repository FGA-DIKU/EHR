from corebehrt.azure import util
from corebehrt.azure.util.config import load_config
from corebehrt.azure.main.helpers import parse_pair_args


def add_parser(subparsers) -> None:
    """
    Add the job subparser
    """
    parser = subparsers.add_parser("job", help="Run a single job.")
    parser.add_argument(
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
    parser.add_argument(
        "COMPUTE",
        type=str,
        help="Compute target to use.",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        help="Path to configuration file. Default is file from repo.",
    )
    parser.add_argument(
        "-e",
        "--experiment",
        type=str,
        default="corebehrt_runs",
        help="Experiment to run the job in.",
    )
    parser.add_argument(
        "-o",
        "--register_output",
        type=str,
        action="append",
        default=[],
        help="If an output should be registered, provide a name for the Azure asset using the format '--register_output <input>=<name>'.",
    )
    parser.add_argument(
        "-lsm",
        "--log_system_metrics",
        action="store_true",
        default=False,
        help="If set, system metrics such as CPU, GPU and memory usage are logged in Azure.",
    )
    parser.set_defaults(func=create_and_run_job)


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
