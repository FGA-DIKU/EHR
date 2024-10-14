from corebehrt.azure import util

from corebehrt.main import pretrain
import argparse

INPUTS = {"data": {"type": "uri_folder", "key": "paths.data_path"}}
OUTPUTS = {"model": {"type": "mlflow_model", "key": "paths.output_path"}}


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
    util.prepare_config("pretrain", INPUTS, OUTPUTS)
    # Run command
    pretrain.main_train(util.AZURE_CONFIG_FILE)
