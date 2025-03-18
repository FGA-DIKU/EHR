from corebehrt.azure import util

INPUTS = {
    "data": {"type": "uri_folder"},
    "vocabulary": {"type": "uri_folder", "optional": True},
    "code_mapping": {"type": "uri_file", "optional": True},
}
OUTPUTS = {
    "tokenized": {"type": "uri_folder"},
    "features": {"type": "uri_folder"},
}


if __name__ == "__main__":
    from corebehrt.main import create_data

    util.run_main(create_data.main_data, INPUTS, OUTPUTS)
