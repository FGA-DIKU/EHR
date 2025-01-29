from corebehrt.azure import util
from corebehrt.main import select_cohort

INPUTS = {
    "patients_info": {"type": "uri_folder"},
    "initial_pids": {"type": "uri_folder", "optional": True},
    "exposure": {"type": "uri_folder", "optional": True},
    "outcome": {"type": "uri_folder"},
}
OUTPUTS = {"cohort": {"type": "uri_folder"}}


def job(config, compute=None, register_output=dict()):
    return util.setup_job(
        "select_cohort",
        inputs=INPUTS,
        outputs=OUTPUTS,
        config=config,
        compute=compute,
        register_output=register_output,
    )


if __name__ == "__main__":
    # Parse args and update config
    util.prepare_config(INPUTS, OUTPUTS)
    # Run command
    select_cohort.main_data(util.AZURE_CONFIG_FILE)
