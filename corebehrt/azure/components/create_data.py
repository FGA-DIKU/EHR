from corebehrt.azure import util

INPUTS = {"data": {"type": "uri_folder"}}
OUTPUTS = {
    "tokenized": {"type": "uri_folder"},
    "features": {"type": "uri_folder"},
}


if __name__ == "__main__":
<<<<<<< HEAD
    from corebehrt.main import create_data

=======
>>>>>>> 3dcaa31 (Azure logging (#137))
    util.run_main(create_data.main_data, INPUTS, OUTPUTS)
