from corebehrt.azure.util import job

INPUTS = {"data": {"type": "uri_folder"}}
OUTPUTS = {
    "features": {"type": "uri_folder"},
    "tokenized": {"type": "uri_folder"},
}


if __name__ == "__main__":
    from corebehrt.main import create_data

    job.run_main("create_data", create_data.main_data, INPUTS, OUTPUTS)
