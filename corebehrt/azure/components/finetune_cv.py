from corebehrt.azure import util

INPUTS = {
    "prepared_data": {"type": "uri_folder"},
    "test_pids": {"type": "uri_file", "optional": True},
    "cohort": {"type": "uri_folder", "optional": True},
    "pretrain_model": {
        "type": "uri_folder",
        "optional": True,
    },
    "restart_model": {
        "type": "uri_folder",
        "optional": True,
    },
}
OUTPUTS = {"model": {"type": "uri_folder"}}


if __name__ == "__main__":
    from corebehrt.main import finetune_cv

    util.run_main(finetune_cv.main_finetune, INPUTS, OUTPUTS)
