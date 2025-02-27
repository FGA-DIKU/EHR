from corebehrt.azure import util

INPUTS = {"data": {"type": "uri_folder"}}
OUTPUTS = {
    "tokenized": {"type": "uri_folder"},
    "features": {"type": "uri_folder"},
}


if __name__ == "__main__":
    util.run_main(create_data.main_data, INPUTS, OUTPUTS)
