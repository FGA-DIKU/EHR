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


def start_run():
    """
    Starts an mlflow run. Used in the Azure wrapper and should
    not in general be used elsewhere.
    """
    if is_mlflow_available():
        mlflow.start_run()


def end_run():
    """
    Ends an mlflow run. Used in the Azure wrapper and should
    not in general be used elsewhere.
    """
    if is_mlflow_available():
        mlflow.end_run()


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

    :param name: Name of the metric.
    :param value: Value of the metric.
    :param step: Step for the metric (used for plotting graphs).
    """
    if is_mlflow_available():
        mlflow.log_metric(*args, **kwargs)
