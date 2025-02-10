from corebehrt.azure import util
from corebehrt.main import create_data
from corebehrt.azure import logging

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
    # Start MLFlow run
    logging.start_run()

    logging.log_metric("test_metric", 42)

    # Parse args and update config
    util.prepare_config(INPUTS, OUTPUTS)

    # Run command
    create_data.main_data(util.AZURE_CONFIG_FILE)

    # End MLFlow run
    logging.end_run()
