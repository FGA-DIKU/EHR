from corebehrt.azure import util

INPUTS = {
    "patients_info": {"type": "uri_file"},
    "initial_pids": {"type": "uri_file", "optional": True},
    "exclude_pids": {"type": "uri_file", "optional": True},
    "exposure": {"type": "uri_file", "optional": True},
    "outcome": {"type": "uri_file"},
}
OUTPUTS = {"cohort": {"type": "uri_folder"}}


if __name__ == "__main__":
    from corebehrt.main import select_cohort

    util.run_main(select_cohort.main_select_cohort, INPUTS, OUTPUTS)
