"""
Base classes for pipeline definitions.
"""

import argparse
from dataclasses import dataclass
from typing import Callable, Optional, List


@dataclass
class PipelineMeta:
    """
    Defines pipeline metadata and CLI arguments in a structured, reusable way.

    Using PipelineMeta allows each pipeline to declare its name, help text, and
    input arguments (as PipelineArg objects) in one place.
    """

    name: str
    help: str
    inputs: List["PipelineArg"]

    def add_to_parser(self, parser: argparse.ArgumentParser):
        group = parser.add_argument_group("Input Data Arguments")
        for arg in self.inputs:
            group.add_argument(
                f"--{arg.name}",
                required=arg.required,
                help=arg.help,
                type=arg.type,
                default=arg.default,
                choices=arg.choices,
                action=arg.action,
                nargs=arg.nargs,
            )


@dataclass
class PipelineArg:
    """
    Describes a single pipeline CLI argument and its properties.
    Is used to add the argument to the parser.
    """

    name: str
    help: str
    required: bool = False
    type: Optional[Callable] = str
    default: Optional[str] = None
    choices: Optional[List[str]] = None
    action: Optional[str] = None
    nargs: Optional[str] = None
