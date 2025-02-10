from corebehrt.azure.util import is_azure_available

MLFLOW_AVAILABLE = False

try:
    import mlflow

    MLFLOW_AVAILABLE = True
except:
    pass


def is_mlflow_available():
    global MLFLOW_AVAILABLE
    return is_azure_available() and MLFLOW_AVAILABLE


def start_run():
    if is_mlflow_available():
        mlflow.start_run()


def end_run():
    if is_mlflow_available():
        mlflow.end_run()


def autolog(*args, **kwargs):
    if is_mlflow_available():
        mlflow.autolog(*args, **kwargs)


def log_metric(*args, **kwargs):
    if is_mlflow_available():
        mlflow.log_metric(*args, **kwargs)
