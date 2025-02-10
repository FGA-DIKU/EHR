from corebehrt.azure import util
from corebehrt.main import create_outcomes

INPUTS = {
    "data": {"type": "uri_folder"},
    "features": {"type": "uri_folder"},
}
OUTPUTS = {"outcomes": {"type": "uri_folder"}}


def job(config, compute=None, register_output=dict()):
    return util.setup_job(
        "create_outcomes",
        inputs=INPUTS,
        outputs=OUTPUTS,
        config=config,
        compute=compute,
        register_output=register_output,
    )


if __name__ == "__main__":
    util.run_main(create_outcomes.main_data, INPUTS, OUTPUTS)
