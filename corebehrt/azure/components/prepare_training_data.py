from corebehrt.azure.util import job

INPUTS = {
    "features": {"type": "uri_folder"},
    "tokenized": {"type": "uri_folder"},
    "cohort": {"type": "uri_folder", "optional": True},
    "outcomes": {"type": "uri_folder", "optional": True},
    "restart_model": {
        "type": "uri_folder",
        "optional": True,
    },
}
OUTPUTS = {"prepared_data": {"type": "uri_folder"}}


if __name__ == "__main__":
    from corebehrt.main import prepare_training_data

    job.run_main(
        "prepare_training_data",
        prepare_training_data.main_prepare_data,
        INPUTS,
        OUTPUTS,
    )
