from corebehrt.azure import util
from corebehrt.main import create_data
import argparse

INPUTS = {
    "features": {"type": "uri_folder"},
    "tokenized": {"type": "uri_folder"},
    "predefined_splits": {
        "type": "uri_folder",
        "optional": True,
    },
    "pretrain_model": {
        "type": "mlflow_model",
        "optional": True,
    },
    "restart_model": {
        "type": "mlflow_model",
        "optional": True,
    },
    "outcome": {"type": "uri_file"},
    "exposure": {"type": "uri_file"},
}
OUTPUTS = {"model": {"type": "mlflow_model"}}


def job(config, compute=None, register_output=dict()):
    return util.setup_job(
        "create_data",
        inputs=INPUTS,
        outputs=OUTPUTS,
        config=config,
        compute=compute,
        register_output=register_output,
    )


if __name__ == "__main__":
    # Parse args and update config
    util.prepare_config("create_data", INPUTS, OUTPUTS)
    # Run command
    create_data.main_data(util.AZURE_CONFIG_FILE)
