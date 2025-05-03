"""
Command-line interface for running CoreBEHRT pipelines on Azure.

This module provides the CLI entry point for running multi-step pipelines
using the Azure ML SDK. It dynamically constructs argument parsers for each
registered pipeline, based on their metadata (see PipelineMeta and PipelineArg
in corebehrt.azure.pipelines.base), and dispatches execution to the appropriate
pipeline definition.

How to use:
    This module is invoked via the command line as part of the corebehrt.azure
    package, for example:

        python -m corebehrt.azure pipeline <PIPELINE_NAME> [pipeline-args] [common-args]

    where <PIPELINE_NAME> is one of the registered pipelines (e.g., E2E, FINETUNE),
    and [pipeline-args] are the required/optional arguments for that pipeline,
    as defined in its PipelineMeta.

Requirements:
    - Each pipeline must be registered in PIPELINE_REGISTRY (see corebehrt.azure.pipelines).
    - Each pipeline must define its arguments using PipelineMeta and PipelineArg.
    - The Azure ML environment and credentials must be set up as described in the project README.

This module is distinct from 'job.py', which runs single-step jobs. Here, pipelines
are orchestrations of multiple jobs/components, with their own input/output wiring
and configuration.

See the project README and the pipelines/ directory for more details and examples.
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

    # Add pipeline-specific arguments
    pipeline.add_to_parser(parser)

    # Add common arguments
    add_common_arguments(parser)

    # Set handler function with the pipeline type
    parser.set_defaults(
        func=lambda args, _name=pipeline.name: run_pipeline(_name, args)
    )


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
    pipeline_meta = PIPELINE_REGISTRY_DICT[pipeline_type]

    # Create input paths dictionary from all defined PipelineArgs
    input_paths = {}
    for arg in pipeline_meta.inputs:
        value = getattr(args, arg.name, None)
        if value is not None:
            input_paths[arg.name] = value

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
