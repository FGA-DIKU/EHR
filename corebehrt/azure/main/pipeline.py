from corebehrt.azure import util
from corebehrt.azure.main.helpers import parse_pair_args


def add_parser(subparsers) -> None:
    """
    Add the pipeline subparser
    """
    parser = subparsers.add_parser("pipeline", help="Run a pipeline job.")
    parser.add_argument("PIPELINE", type=str, choices={"E2E"}, help="Pipeline to run.")
    parser.add_argument("DATA", type=str, help="Raw input data.")
    parser.add_argument(
        "COMPUTE",
        type=str,
        default=None,
        nargs="?",
        help="Default compute target to use. If not set, compute targets must be specified for all components.",
    )
    parser.add_argument(
        "CONFIG_DIR",
        type=str,
        default=None,
        nargs="?",
        help="Path to folder with configs. If not set, configs must be specified for all components.",
    )
    parser.add_argument(
        "-cp",
        "--compute",
        type=str,
        action="append",
        default=[],
        help="Compute target to use for a specific pipeline step. Use: '-cp <job_name>=<compute>'.",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        action="append",
        default=[],
        help="Path to configuration file for each component. Use: '-c <job_name>=<path-to-cfg>'. If not set, default is file from repo.",
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
        help="If an output from any step in the pipeline should be registered, provide a name for the Azure asset using the format '--register_output <job_name>.<input>=<name>'.",
    )
    parser.add_argument(
        "-lsm",
        "--log_system_metrics",
        action="store_true",
        default=False,
        help="If set, system metrics such as CPU, GPU and memory usage are logged in Azure.",
    )
    parser.set_defaults(func=create_and_run_pipeline)


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
