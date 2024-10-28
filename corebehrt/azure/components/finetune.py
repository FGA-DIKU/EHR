from corebehrt.azure import util
from corebehrt.main import finetune_cv

INPUTS = {
    "features": {"type": "uri_folder"},
    "tokenized": {"type": "uri_folder"},
    "predefined_splits": {
        "type": "uri_folder",
        "optional": True,
    },
    "pretrain_model": {
        "type": "uri_folder",
        "optional": True,
    },
    "restart_model": {
        "type": "uri_folder",
        "optional": True,
    },
    "outcome": {"type": "uri_file"},
    "exposure": {"type": "uri_file"},
}
OUTPUTS = {"model": {"type": "uri_folder"}}


def job(config, compute=None, register_output=dict()):
    return util.setup_job(
        "finetune_cv",
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
    finetune_cv.main_finetune(util.AZURE_CONFIG_FILE)
