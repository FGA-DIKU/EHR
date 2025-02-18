from corebehrt.azure import util
from corebehrt.main import finetune_cv

INPUTS = {
    "features": {"type": "uri_folder"},
    "tokenized": {"type": "uri_folder"},
    "cohort": {"type": "uri_folder"},
    "test_pids": {"type": "uri_file", "optional": True},
    "pretrain_model": {
        "type": "uri_folder",
        "optional": True,
    },
    "restart_model": {
        "type": "uri_folder",
        "optional": True,
    },
    "outcome": {"type": "uri_file"},
}
OUTPUTS = {"model": {"type": "uri_folder"}}


def job(config, compute=None, register_output=dict()):
    return util.setup_job(
        "finetune",
        inputs=INPUTS,
        outputs=OUTPUTS,
        config=config,
        compute=compute,
        register_output=register_output,
    )


if __name__ == "__main__":
    util.run_main(finetune_cv.main_finetune, INPUTS, OUTPUTS)
