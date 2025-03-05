from corebehrt.azure import util

INPUTS = {
    "prepared_data": {"type": "uri_folder"},
    "cohort": {"type": "uri_folder", "optional": True},
    "restart_model": {
        "type": "uri_folder",
        "optional": True,
    },
}
OUTPUTS = {"model": {"type": "uri_folder"}}


if __name__ == "__main__":
    from corebehrt.main import pretrain

    util.run_main(pretrain.main_train, INPUTS, OUTPUTS)
