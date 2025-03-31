from contextlib import contextmanager
import time

MLFLOW_AVAILABLE = False
MLFLOW_CLIENT = None

try:
    # Try to import mlflow and set availability flag
    import mlflow
    from mlflow.tracking import MlflowClient
    from mlflow.entities import Metric

    MLFLOW_AVAILABLE = True
    MLFLOW_CLIENT = MlflowClient()
except:
    pass


def is_mlflow_available() -> bool:
    """
    Checks if mlflow module is available.
    """
    global MLFLOW_AVAILABLE
    return MLFLOW_AVAILABLE


def start_run(name: str = None, nested: bool = False, log_system_metrics: bool = False):
    """
    Starts an mlflow run. Used in the Azure wrapper and should
    not in general be used elsewhere.

    :param name: Name of run
    :param nested: If the run should be nested.
    :param log_system_metrics: If enabled, log system metrics (CPU/GPU/mem).
    """
    if is_mlflow_available():
        run = mlflow.start_run(
            run_name=name, nested=nested, log_system_metrics=log_system_metrics
        )
    else:
        # Return a dummpy context manager so as to not raise an
        # error if mlflow is not available
        @contextmanager
        def dummy_cm():
            yield None

        run = dummy_cm()

    return run


def end_run():
    """
    Ends an mlflow run. Used in the Azure wrapper and should
    not in general be used elsewhere.
    """
    if is_mlflow_available():
        mlflow.end_run()


def setup_metrics_dir(name: str):
    """
    Shorthand for starting a sub-run where metrics will be logged.
    Use as context manager.

    :param name: Name of azure "sub-dir"/metrics pane.
    """
    return start_run(name=name, nested=True)


#
# Simple wrapper functions below. Review full args at:
# https://www.mlflow.org/docs/latest/python_api/mlflow.html
#


def log_metric(*args, **kwargs):
    """
    Logs a metric to the job (if mlflow is available).
    Parameters are the same as for mlflow.log_metric.

    Important parameters:

    :param key: Name of the metric.
    :param value: Value of the metric.
    :param step: Step for the metric (used for plotting graphs).
    """
    if is_mlflow_available():
        mlflow.log_metric(*args, **kwargs)


def log_metrics(*args, **kwargs):
    """
    Log multiple metrics

    :param metrics: dict of metrics.
    :param step: Step for the metric (used for plotting graphs).
    """
    if is_mlflow_available():
        mlflow.log_metrics(*args, **kwargs)


def log_param(*args, **kwargs):
    """
    Log a parameter

    :param key: Name of the param.
    :param value: Value of the param.
    """
    if is_mlflow_available():
        mlflow.log_param(*args, **kwargs)


def log_params(*args, **kwargs):
    """
    Log multiple parameters.

    :param params: dict of parameters.
    """
    if is_mlflow_available():
        mlflow.log_params(*args, **kwargs)


def log_image(*args, **kwargs):
    """
    Log an image

    :param image: e.g. numpy array or PIL image.
    :param artifact_file: filename to save image under.
    """
    if is_mlflow_available():
        mlflow.log_image(*args, **kwargs)


def log_figure(*args, **kwargs):
    """
    Log a figure (e.g. matplotlib)

    :param figure: e.g. matplotlib figure.
    :param artifact_file: filename to save image under.
    """
    if is_mlflow_available():
        mlflow.log_figure(*args, **kwargs)


def log_batch(*args, **kwargs):
    """
    Log a batch of metrics

    :param metrics: metrics list
    """
    if is_mlflow_available():
        global MLFLOW_CLIENT
        run = mlflow.active_run()
        MLFLOW_CLIENT.log_batch(*args, run_id=run.info.run_id, **kwargs)


def metric(name, value, step):
    if is_mlflow_available():
        timestamp = int(time.time() * 1000)
        return Metric(name, value, timestamp, step)
    else:
        return (name, value)
