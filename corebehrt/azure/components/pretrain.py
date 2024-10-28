from corebehrt.azure import util

from corebehrt.main import pretrain
import argparse

INPUTS = {
    "features": {"type": "uri_folder"},
    "tokenized": {"type": "uri_folder"},
    "predefined_splits": {
        "type": "uri_folder",
        "optional": True,
    },
    "restart_model": {
        "type": "mlflow_model",
        "optional": True,
    },
}
OUTPUTS = {"model": {"type": "mlflow_model"}}


def job(config, compute=None, register_output=dict()):
    return util.setup_job(
        "pretrain",
        inputs=INPUTS,
        outputs=OUTPUTS,
        config=config,
        compute=compute,
        register_output=register_output,
    )


if __name__ == "__main__":
    # Parse args and update config
    util.prepare_config(INPUTS, OUTPUTS)
    # Run command
    pretrain.main_train(util.AZURE_CONFIG_FILE)
