import os
import yaml

from .util import check_azure, ml_client

TMP_CONDA_FILE = ".cb_conda.yaml"
REQUIREMENTS_FILE = "requirements.txt"
ADDITIONAL_DEPENDENCIES = ["mlflow", "azureml-mlflow", "pynvml"]


def build():
    """
    Build the CoreBEHRT environment.
    """
    check_azure()

    # Get MLClient object
    mlc = ml_client()

    # Create config file
    create_conda_cfg()

    # Create environment and start build
    from azure.ai.ml.entities import Environment

    env = Environment(
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
        conda_file=TMP_CONDA_FILE,
        name="CoreBEHRT",
        description="Environment created by the CoreBEHRT package.",
    )
    mlc.environments.create_or_update(env)

    # Clean-up
    remove_conda_cfg()


def create_conda_cfg() -> None:
    """
    Create a conda config based on a requirements.txt

    The config file is save to TMP_CONDA_FILE.
    The requirements are read from REQUIREMENTS_FILE
    """
    # Check that requirements is available
    if not os.path.exists(REQUIREMENTS_FILE):
        raise FileNotFoundError("requirements.txt not found!")

    # Read requirements
    with open(REQUIREMENTS_FILE, "r") as f:
        lines = f.readlines()

    # Add lines
    lines += ADDITIONAL_DEPENDENCIES

    # Create and save conda config
    cfg = {
        "name": "CoreBEHRT",
        "channels": ["defaults", "conda-forge", "anaconda", "pytorch"],
        "dependencies": ["python>=3.12.0,<3.13.0", "pip", {"pip": lines}],
    }
    with open(TMP_CONDA_FILE, "w") as f:
        f.write(yaml.dump(cfg))


def remove_conda_cfg() -> None:
    """
    Clean-up the temporary conda file
    """
    os.remove(TMP_CONDA_FILE)
