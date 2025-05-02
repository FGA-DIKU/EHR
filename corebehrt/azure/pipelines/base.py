# corebehrt/azure/pipelines/base.py
import argparse
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional


@dataclass
class PipelineMeta:
    name: str
    help: str
    required_inputs: Dict[str, Dict]
    helper_inputs: Dict[str, Dict] = field(default_factory=dict)
    add_args: Optional[Callable] = None  # function to add args to parser

    def add_to_parser(self, parser: argparse.ArgumentParser):
        group = parser.add_argument_group("Input Data Arguments")
        for arg, meta in self.required_inputs.items():
            group.add_argument(f"--{arg}", required=True, help=meta["help"])
        for arg, meta in self.helper_inputs.items():
            group.add_argument(f"--{arg}", required=False, help=meta["help"])
        if self.add_args:
            self.add_args(parser)
