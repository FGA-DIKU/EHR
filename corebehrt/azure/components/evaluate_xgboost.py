from corebehrt.azure.util import job

INPUTS = {
    "test_data_dir": {"type": "uri_folder"},
    "model": {"type": "uri_folder"},
    "folds_dir": {"type": "uri_folder"},
}
OUTPUTS = {"predictions": {"type": "uri_folder"}}

if __name__ == "__main__":
    from corebehrt.main import evaluate_xgboost

    job.run_main("evaluate_xgboost", evaluate_xgboost.main_evaluate, INPUTS, OUTPUTS)
