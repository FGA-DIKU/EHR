from corebehrt.azure import util
from corebehrt.main import create_outcomes
import argparse

INPUTS = {
    "data": {"type": "uri_folder"},
    "features": {"type": "uri_folder"},
}
OUTPUTS = {"outcomes": {"type": "uri_folder"}}


def job(config, compute=None, register_output=dict()):
    return util.setup_job(
        "create_outcomes",
        inputs=INPUTS,
        outputs=OUTPUTS,
        config=config,
        compute=compute,
        register_output=register_output,
    )


if __name__ == "__main__":
    # Parse args and update config
    util.prepare_config("create_outcomes", INPUTS, OUTPUTS)
    # Run command
    create_outcomes.main_data(util.AZURE_CONFIG_FILE)
