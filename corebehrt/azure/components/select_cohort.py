from corebehrt.azure.util import job

INPUTS = {
    "features": {"type": "uri_folder"},
    #"initial_pids": {"type": "uri_file", "optional": True},
    "tokenized": {"type": "uri_folder"},
    "exclude_pids": {"type": "uri_file", "optional": True},
    "outcomes": {"type": "uri_folder"},
}
OUTPUTS = {"cohort": {"type": "uri_folder"}}


if __name__ == "__main__":
    from corebehrt.main import select_cohort

    job.run_main("select_cohort", select_cohort.main_select_cohort, INPUTS, OUTPUTS)
