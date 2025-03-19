import sys
from corebehrt.azure.util.config import load_config
import logging
import time


def evaluate_run(run_id: str, job_name: str, test_cfg_file: str):  # noqa: F821
    """
    If given a MLFlow Run object, evaluates the run according
    to the given job_name and test configuration file.
    """

    logger = logging.getLogger("test")

    ## Setup
    try:
        run = mlflow.get_run(run_id)
    except:
        logger.error("Could not load run!")
        return

    # Check that cfg file exists
    try:
        cfg = load_config(test_cfg_file)
    except:
        logger.error(
            "Could not load test configuration file - no evaluation will be performed"
        )
        return

    # Get sub-config for this job
    if not (cfg := cfg.get(job_name, False)):
        logger.warning(
            f"No sub-config found for {job_name} - no evaluation will be performed for {job_name}"
        )
        return

    ## Test evaluation
    results = []

    # Run time
    if max_run_time := cfg.get("max_run_time", False):
        results.append(perform_time_test(run, max_value=max_run_time))

    # Metrics test
    metrics = cfg.get("metrics", {})
    for metric, metric_cfg in metrics.items():
        results.append(
            perform_metric_test(
                run,
                metric,
                min_value=metric_cfg.get("min"),
                max_value=metric_cfg.get("max"),
            )
        )

    ## Finish
    if all(results):
        logger.info("All tests passed!")
    else:
        num_failed = len(results) - sum(results)
        msg = f"{num_failed} test(s) failed!"
        logger.error(msg)
        print(msg, file=sys.stderr)
        if cfg.get("on_fail") == "raise":
            raise Exception(msg)


def perform_time_test(run, max_value: int) -> bool:
    end_time = run.info.end_time or int(time.time() * 1000)
    run_time = (end_time - run.info.start_time) // 1000
    return log_test_result(
        "Run time",
        run_time <= max_value,
        f"Actual run time: {run_time}, threshold: {max_value}",
    )


def perform_metric_test(
    run, metric: str, min_value: float = None, max_value: float = None
) -> bool:
    if metric not in run.data.metrics:
        return log_rest_result(f"{metric}", False, f"Metric not found!")

    metric_value = run.data.metrics.get(metric)

    min_ok = True
    max_ok = True
    if min_value:
        min_ok = log_test_result(
            f"{metric} [minimum]", metric_value >= min_value, f"{metric}>{min_value}"
        )
    if max_value:
        max_ok = log_test_result(
            f"{metric} [maximum]", metric_value <= max_value, f"{metric}<{max_value}"
        )
    return min_ok and max_ok


def log_test_result(test_name: str, ok: bool, msg: str = "") -> bool:
    logger = logging.getLogger("test")

    if ok:
        msg = f"{test_name} test passed. " + msg
        logger.info(msg)
    else:
        msg = f"{test_name} test failed. " + msg
        logger.warning(msg)
    return ok
