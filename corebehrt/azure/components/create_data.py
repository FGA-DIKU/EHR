from corebehrt.azure import util
from corebehrt.main import create_data

INPUTS = {"data": {"type": "uri_folder"}}
OUTPUTS = {
    "tokenized": {"type": "uri_folder"},
    "features": {"type": "uri_folder"},
}


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
    util.prepare_config(INPUTS, OUTPUTS)
    # Run command
    create_data.main_data(util.AZURE_CONFIG_FILE)
