from corebehrt.azure.util import job

INPUTS = {
    "test_data_dir": {"type": "uri_folder"},
    "model_path": {"type": "uri_folder"},
    "processed_data": {"type": "uri_folder"},
}
OUTPUTS = {"predictions": {"type": "uri_folder"}}

if __name__ == "__main__":
    from corebehrt.main import evaluate_finetune

    job.run_main("evaluate_finetune", evaluate_finetune.main_evaluate, INPUTS, OUTPUTS)
