from corebehrt.azure.util import job

INPUTS = {
    "prepared_data": {"type": "uri_folder"},
    "test_pids": {"type": "uri_file", "optional": True},
}
OUTPUTS = {"model": {"type": "uri_folder"}}


if __name__ == "__main__":
    from corebehrt.main import xgboost_cv

    job.run_main("xgboost_cv", xgboost_cv.main_xgboost, INPUTS, OUTPUTS)
