from corebehrt.azure import util

INPUTS = {
    "features": {"type": "uri_folder"},
    "tokenized": {"type": "uri_folder"},
    "cohort": {"type": "uri_folder", "optional": True},
    "outcome": {"type": "uri_file", "optional": True},
    "restart_model": {
        "type": "uri_folder",
        "optional": True,
    },
}
OUTPUTS = {"prepared_data": {"type": "uri_folder"}}



if __name__ == "__main__":
    util.run_main(prepare_training_data.main_prepare_data, INPUTS, OUTPUTS)
