import argparse
import logging
import os
import uuid
from os.path import join, split
from shutil import rmtree, copyfile
from typing import Tuple

from corebehrt.common.config import Config, load_config

logger = logging.getLogger(__name__)  # Get the logger for this module

CHECKPOINTS_DIR = "checkpoints"

DATA_CFG = "data_config.yaml"
OUTCOMES_CFG = "outcomes_config.yaml"
PRETRAIN_CFG = "pretrain_config.yaml"
FINETUNE_CFG = "finetune_config.yaml"


def get_args(default_config_name, default_run_name=None):
    """Get command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default=default_config_name)
    parser.add_argument(
        "--run_name",
        type=str,
        default=(
            default_run_name if default_run_name else default_config_name.split(".")[0]
        ),
    )
    return parser.parse_args()


class DirectoryPreparer:
    """Prepares directories for training and evaluation."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

        # Check that paths exist
        if not hasattr(cfg, "paths"):
            raise ValueError("paths must be set in configuration file.")

        # Set logging defaults
        if not hasattr(cfg, "logging") or not hasattr(cfg.logging, "level"):
            cfg.logging.level = logging.INFO

    def setup_logging(
        self, log_name: str, log_dir: str = None, log_level: str = None
    ) -> None:
        log_dir = (
            log_dir
            or self.cfg.logging.get("path")
            or self.cfg.paths.get("root")
            or "./logs"
        )
        log_level = log_level or self.cfg.logging.level

        """Sets up the logger."""
        os.makedirs(log_dir, exist_ok=True)
        logging.basicConfig(
            filename=join(log_dir, f"{log_name}.log"),
            level=log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

    def get_config_path(self, directory: str, name: str = None) -> str:
        if name is None:
            # Default name based on source
            name = {
                "features": DATA_CFG,
                "tokenized": DATA_CFG,
                "outcomes": OUTCOMES_CFG,
                "model": PRETRAIN_CFG,
            }[directory]

        path = self.check_path(directory, use_root=True)
        return join(path, name)

    def get_config(self, source: str, name: str = None) -> Config:
        try:
            return load_config(self.get_config_path(source, name))
        except:
            return None

    def write_config(self, target: str, source: str = None, name: str = None) -> bool:
        # Path to target
        target_path = self.get_config_path(target, name=name)

        if source is not None:
            # Copy file from source
            source_path = self.get_config_path(source, name=name)
            copyfile(source_path, target_path)

        else:
            # Save the current file to target
            self.cfg.save_to_yaml(target_path)

    def check_path(self, target: str, use_root: bool = True) -> str:
        if not hasattr(self.cfg.paths, target):
            if not use_root:
                raise ValueError(f"paths.{target} must be set")
            if not hasattr(self.cfg.paths, "root"):
                raise ValueError(
                    f"paths.root must be set if paths.{target} is not set."
                )
            self.cfg.paths[target] = join(self.cfg.paths.root, target)
        return self.cfg.paths[target]

    def check_exist(self, target: str, use_root: bool = True) -> str:
        path = self.check_path(target, use_root=use_root)
        if not os.path.exists(path):
            raise ValueError(f"paths.{target} (= '{path}') does not exist.")
        return path

    def check_file(self, target: str) -> str:
        path = self.check_exist(target, use_root=False)
        if not os.path.isfile(path):
            raise ValueError(f"paths.{target} (= '{path}') is not a file.")
        return path

    def check_directory(self, target: str, use_root: bool = True) -> str:
        path = self.check_exist(target, use_root=False)
        if not os.path.isdir(path):
            raise ValueError(f"paths.{target} (= '{path}') is not a directory.")
        return path

    def create_directory(self, target_dir: str, clear: bool = False) -> str:
        path = self.check_path(target_dir, use_root=True)
        os.makedirs(path, exist_ok=True)

        if clear:
            files = os.listdir(path)
            if len(files) > 0:
                logger = logging.getLogger(__name__)
                logger.warning(
                    f"Directory '{path}' is not empty. {len(files)} will be deleted."
                )
                for file in files:
                    file_path = join(path, file)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                    else:
                        rmtree(file_path)
        return path

    def create_run_directory(
        self, target_dir: str, base_dir: str = None, run_id: str = None
    ) -> str:
        ## Output
        if not hasattr(self.cfg.paths, target_dir) and hasattr(
            self.cfg.paths, base_dir
        ):
            # Generate a run name / name of run directory
            run_id = run_id or self.generate_run_id()

            # Set target dir
            self.cfg.paths[target_dir] = join(self.cfg.paths[base_dir], run_id)

            # When generating a run dir, point logs to that dir
            self.setup_logging("run", self.cfg.paths[target_dir])

        # Create the run directory
        run_dir = self.create_directory(target_dir)

        # Create the required sub-directory
        os.makedirs(join(run_dir, CHECKPOINTS_DIR), exist_ok=True)

        return run_dir

    def setup_create_data(self) -> None:
        # Setup logging
        self.setup_logging("create_data")

        # Validate and create directories
        self.check_directory("data", use_root=False)
        self.create_directory("features")
        self.create_directory("tokenized", clear=True)

        # Write config in output directories.
        self.write_config("features", name=DATA_CFG)
        self.write_config("tokenized", name=DATA_CFG)

    def setup_create_outcomes(self) -> None:
        # Setup logging
        self.setup_logging("create_outcomes")

        # Validate and create directories
        self.check_directory("data", use_root=False)
        self.check_directory("features")
        self.create_directory("outcomes")

        # Write config in output directory.
        self.write_config("outcomes", source="features", name=DATA_CFG)
        self.write_config("outcomes", name=OUTCOMES_CFG)

    def setup_pretrain(self) -> None:
        # Setup logging
        self.setup_logging("pretrain")

        # Validate and create directories
        self.check_directory("features")
        self.check_directory("tokenized")
        self.create_run_directory("model", base_dir="runs")

        # Write config in output directory.
        self.write_config("model", name=DATA_CFG)
        self.write_config("model", name=PRETRAIN_CFG)

    def setup_finetune(self) -> None:
        # Setup logging
        self.setup_logging("finetune")

        # Validate and create directories
        self.check_directory("features")
        self.check_directory("tokenized")
        self.check_directory("pretrain_model")
        self.check_file("outcome")
        self.check_file("exposure")
        self.create_run_directory(
            "model", base_dir="runs", run_id=self.generate_finetune_run_id()
        )

        # Write config in output directory.
        self.write_config("model", source="features", name=DATA_CFG)
        self.write_config("model", source="pretrain_model", name=PRETRAIN_CFG)
        self.write_config("model", name=FINETUNE_CFG)

        # Add pretrain info to config
        data_cfg = self.get_config("model", name=DATA_CFG)
        self.cfg.paths.data = data_cfg.paths.data
        if "tokenized" not in self.cfg.paths:
            logger.info("Tokenized dir not in config. Adding from pretrain config.")
            self.cfg.paths.tokenized = data_cfg.paths.tokenized

    #
    # Directory naming generators
    #
    def generate_run_id(self) -> str:
        if hasattr(self.cfg.paths, "run_name"):
            return self.cfg.paths.run_name

        return uuid.uuid4().hex

    def generate_finetune_run_id(self) -> str:
        """
        Constructs the name of the finetune model directory.
        Based on the outcome type, the censor type, and the number of hours pre- or post- outcome.
        """
        suffix = (
            self.cfg.paths.run_name
            if hasattr(self.cfg.paths, "run_name")
            else uuid.uuid4().hex
        )

        outcome_name = self.get_event_name(self.cfg.paths.outcome)
        censor_name = (
            self.get_event_name(self.cfg.paths.exposure)
            if self.cfg.paths.get("exposure", False)
            else outcome_name
        )
        n_hours_censor = self.cfg.outcome.get("n_hours_censoring", None)
        n_hours_str = (
            DirectoryPreparer.handle_n_hours(n_hours_censor)
            if n_hours_censor is not None
            else "at"
        )
        if self.cfg.outcome.get("index_date", None) is not None:
            censor_name = DirectoryPreparer.handle_index_date(
                self.cfg.outcome.index_date
            )

        run_id = f"finetune_{outcome_name}_censored_{n_hours_str}_{censor_name}"

        n_hours_start_follow_up = self.cfg.outcome.get("n_hours_follow_up", None)
        n_hours_follow_up_str = (
            DirectoryPreparer.handle_n_hours(n_hours_start_follow_up)
            if n_hours_start_follow_up is not None
            else "at"
        )

        return f"{run_id}_followup_start_{n_hours_follow_up_str}_index_date_{suffix}"

    @staticmethod
    def get_event_name(path: str) -> str:
        return split(path)[-1].strip(".csv")

    @staticmethod
    def handle_n_hours(n_hours: int) -> str:
        days = True if abs(n_hours) > 48 else False
        window = int(abs(n_hours / 24)) if days else abs(n_hours)
        days_hours = "days" if days else "hours"
        pre_post = "pre" if n_hours < 0 else "post"
        return f"{window}_{days_hours}_{pre_post}"

    @staticmethod
    def handle_index_date(n_hours: dict) -> str:
        censor_event = [f"{k}{v}" for k, v in n_hours.items() if v is not None]
        return "_".join(censor_event)
