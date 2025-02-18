from corebehrt.azure import util

INPUTS = {
    "tokenized": {"type": "uri_folder"},
    "prepared_data": {"type": "uri_folder"},
    "cohort": {"type": "uri_folder", "optional": True},
    "restart_model": {
        "type": "uri_folder",
        "optional": True,
    },
}
OUTPUTS = {"model": {"type": "uri_folder"}}


if __name__ == "__main__":
<<<<<<< HEAD
    from corebehrt.main import pretrain

=======
>>>>>>> 3dcaa31 (Azure logging (#137))
    util.run_main(pretrain.main_train, INPUTS, OUTPUTS)
