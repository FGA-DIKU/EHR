from corebehrt.azure import util

INPUTS = {
    "data": {"type": "uri_folder"},
    "features": {"type": "uri_folder"},
}
OUTPUTS = {"outcomes": {"type": "uri_folder"}}

if __name__ == "__main__":
    from corebehrt.main import create_outcomes

    util.run_main(create_outcomes.main_data, INPUTS, OUTPUTS)
