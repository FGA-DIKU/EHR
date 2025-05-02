"""
Command-line interface for running CoreBEHRT pipelines on Azure.
"""

import argparse

from corebehrt.azure import util
from corebehrt.azure.main.helpers import parse_pair_args
from corebehrt.azure.pipelines import PIPELINE_REGISTRY
from corebehrt.azure.pipelines.base import PipelineMeta
from corebehrt.azure.pipelines.parser import add_common_arguments

PIPELINE_REGISTRY_DICT = {p.name: p for p in PIPELINE_REGISTRY}


def add_pipeline_parser(
    subparsers: argparse._SubParsersAction, pipeline: PipelineMeta
) -> None:
    """
    Add a parser for a specific pipeline type.

    Args:
        subparsers: The subparsers action to add the parser to
        pipeline_type: The name of the pipeline type
        config: Configuration for this pipeline type
    """
    # Create parser for this pipeline type
    parser: argparse.ArgumentParser = subparsers.add_parser(
        pipeline.name, help=pipeline.help
    )

    # Add input group for required inputs
    input_group = parser.add_argument_group("Input Data Arguments")

    # Add required inputs
    for input_name, help_text in pipeline.required_inputs.items():
        input_group.add_argument(
            f"--{input_name}",
            type=str,
            required=True,
            help=help_text,
        )

    # Add common arguments
    add_common_arguments(parser)

    # Set handler function with the pipeline type
    parser.set_defaults(func=lambda args: run_pipeline(pipeline.name, args))


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    """
    Add the pipeline subparser with subparsers for each pipeline type.

    Args:
        subparsers: The subparsers action to add the pipeline parser to
    """
    # Main pipeline parser
    pipeline_parser: argparse.ArgumentParser = subparsers.add_parser(
        "pipeline", help="Run a pipeline job."
    )

    # Create subparsers for each pipeline type
    pipeline_subparsers = pipeline_parser.add_subparsers(
        dest="pipeline_type",
        help="Type of pipeline to run",
        required=True,
    )

    # Add parsers for each pipeline type
    for pipeline in PIPELINE_REGISTRY:
        add_pipeline_parser(pipeline_subparsers, pipeline)


def run_pipeline(pipeline_type: str, args: argparse.Namespace) -> None:
    """
    Run a pipeline with the specified type and arguments.

    Args:
        pipeline_type: The type of pipeline to run
        args: The parsed command line arguments (from argparse) defined in the pipeline parser
    """
    # Get required inputs for this pipeline type
    required_inputs = PIPELINE_REGISTRY_DICT[pipeline_type].required_inputs

    # Create input paths dictionary
    input_paths = {}
    for input_name in required_inputs:
        input_paths[input_name] = getattr(args, input_name)

    # Process common arguments
    cfg_paths = parse_pair_args(args.config)
    computes = parse_pair_args(args.compute)
    register_output = parse_pair_args(args.register_output)

    # Set default compute if provided
    if args.COMPUTE is not None:
        computes["default"] = args.COMPUTE

    # Create the pipeline
    pl = util.pipeline.create(
        name=pipeline_type,
        input_paths=input_paths,
        computes=computes,
        config_paths=cfg_paths,
        config_dir=args.CONFIG_DIR,
        register_output=register_output,
        log_system_metrics=args.log_system_metrics,
    )

    # Run the pipeline
    util.pipeline.run(pl, args.experiment)
