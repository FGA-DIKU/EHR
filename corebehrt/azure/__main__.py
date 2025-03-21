"""
Command-line utility for starting CoreBEHRT jobs on Azure clusters.
Requires installation of azure-ml-ai python package and a valid Azure workspace.
"""

import sys
import argparse
from corebehrt.azure import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="corebehrt.azure", description="Run corebehrt jobs and pipelines in Azure"
    )

    # Sub-parsers
    subparsers = parser.add_subparsers(dest="call_type")
    main.environment.add_parser(subparsers)
    main.job.add_parser(subparsers)
    main.pipeline.add_parser(subparsers)
    main.test.add_parser(subparsers)

    # Parse args
    args = parser.parse_args()

    if hasattr(args, "func"):
        # Run appropriate main
        args.func(args)
    else:
        parser.print_help()
        sys.exit(1)
