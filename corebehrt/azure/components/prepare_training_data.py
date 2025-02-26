from corebehrt.azure import util
from corebehrt.main import prepare_training_data

INPUTS = {
    "features": {"type": "uri_folder"},
    "tokenized": {"type": "uri_folder"},
    "cohort": {"type": "uri_folder", "optional": True},
    "outcome": {"type": "uri_folder", "optional": True},
    "restart_model": {
        "type": "uri_folder",
        "optional": True,
    },
}
OUTPUTS = {"prepared_data": {"type": "uri_folder"}}


def job(config, compute=None, register_output=dict()):
    return util.setup_job(
        "prepare_training_data",
        inputs=INPUTS,
        outputs=OUTPUTS,
        config=config,
        compute=compute,
        register_output=register_output,
    )


if __name__ == "__main__":
    util.run_main(prepare_training_data.main_prepare_data, INPUTS, OUTPUTS)
