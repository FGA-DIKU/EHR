import sys
from corebehrt.azure.util.config import load_config
import logging
import time


def evaluate_run(run_id: str, job_type: str, test_cfg_file: str):
    """
    Ealuates the run according to the given job_type and test configuration file.

    :param run_id: ID of the run to evaluate.
    :param job_type: Type of the current job.
    :param test_cfg_file: Path to test configuration file.
    """

    logger = logging.getLogger("test")
    logger.info(f"Evaluating run using config file {test_cfg_file}")

    ## Setup
    try:
        import mlflow

        run = mlflow.get_run(run_id)
    except Exception as e:
        logger.error(f"Could not load run with ID {run_id}! Error: {e}")
        return

    # Check that cfg file exists
    try:
        cfg = load_config(test_cfg_file)
    except:
        logger.error(
            "Could not load test configuration file - no evaluation will be performed"
        )
        return

    # Read 'on_fail' from top level config
    on_fail = cfg.get("on_fail")

    # Get sub-config for this job
    if not (cfg := cfg.get(job_type, False)):
        logger.warning(
            f"No sub-config found for {job_type} - no evaluation will be performed for {job_type}"
        )
        return

    ## Test evaluation
    results = []

    # Run time
    if max_run_time := cfg.get("max_run_time", False):
        results.append(perform_time_test(run, max_value=max_run_time))

    # Metrics test
    metrics = cfg.get("metrics", [])
    children = get_children(run_id)
    for metric_cfg in metrics:
        relevant_run = children.get(metric_cfg.get("child"), run)
        results.append(perform_metric_test(relevant_run, metric_cfg))

    ## Finish
    if all(results):
        logger.info("All tests passed!")
    else:
        msg = f"One or more tests failed!"
        logger.error(msg)
        print(msg, file=sys.stderr)
        if on_fail == "raise":
            raise Exception(msg)


def perform_time_test(run, max_value: int) -> int:
    """
    Test that run time is not greater than the threshold.
    """
    end_time = run.info.end_time or int(time.time() * 1000)
    run_time = (end_time - run.info.start_time) // 1000
    return log_test_result(
        "Run time",
        run_time <= max_value,
        f"Actual run time: {run_time}, threshold: {max_value}",
    )


def perform_metric_test(run, metric_cfg: dict) -> bool:
    """
    Test that the given metric satisfies the min/max constraints.
    """
    # Check that type is set
    if not (metric := metric_cfg.get("type", False)):
        return log_test_result("Metric", False, "Metric type missing!")

    # Check that metric exists
    if metric not in run.data.metrics:
        return log_test_result(metric, False, "Metric not found in run!")

    # Get metric and thresholds
    metric_value = run.data.metrics.get(metric)
    min_value = metric_cfg.get("min")
    max_value = metric_cfg.get("max")

    min_ok = True
    max_ok = True
    if min_value:
        min_ok = log_test_result(
            f"{metric} [minimum]",
            metric_value >= min_value,
            f"{metric_value}>{min_value}",
        )
    if max_value:
        max_ok = log_test_result(
            f"{metric} [maximum]",
            metric_value <= max_value,
            f"{metric_value}<{max_value}",
        )

    return min_ok and max_ok


def log_test_result(test_name: str, ok: bool, msg: str = "") -> int:
    """
    Helper for logging result of a test.

    :param test_name: Name of the test
    :param ok: Result of the test.
    :param msg: Additional message.
    """
    logger = logging.getLogger("test")

    if ok:
        msg = f"{test_name} test passed. " + msg
        logger.info(msg)
    else:
        msg = f"{test_name} test failed. " + msg
        logger.warning(msg)
    return ok


def get_children(run_id) -> dict:  # noqa: F821
    import mlflow

    query = f"tags.mlflow.parentRunId = '{run_id}'"
    results = mlflow.search_runs(filter_string=query)
    return {
        r["tags.mlflow.runName"]: mlflow.get_run(r.run_id)
        for _, r in results.iterrows()
    }
