from contextlib import contextmanager

MLFLOW_AVAILABLE = False

try:
    # Try to import mlflow and set availability flag
    import mlflow

    MLFLOW_AVAILABLE = True
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
        return mlflow.start_run(
            run_name=name, nested=nested, log_system_metrics=log_system_metrics
        )
    else:
        # Return a dummpy context manager so as to not raise an
        # error if mlflow is not available
        @contextmanager
        def dummy_cm():
            yield None

        return dummy_cm()


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


def autolog(*args, **kwargs):
    """
    Enables mlflow autologging (if mlflow is available)
    """
    if is_mlflow_available():
        mlflow.autolog(*args, **kwargs)


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
