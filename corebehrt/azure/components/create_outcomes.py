from corebehrt.azure import util

INPUTS = {
    "data": {"type": "uri_folder"},
    "features": {"type": "uri_folder"},
}
OUTPUTS = {"outcomes": {"type": "uri_folder"}}

if __name__ == "__main__":
    util.run_main(create_outcomes.main_data, INPUTS, OUTPUTS)
