MLFLOW_AVAILABLE = False

try:
    import mlflow

    MLFLOW_AVAILABLE = True
except:
    pass


def mlflow_available():
    global MLFLOW_AVAILABLE
    return MLFLOW_AVAILABLE


def start_run():
    if MLFLOW_AVAILABLE:
        mlflow.start_run()


def end_run():
    if MLFLOW_AVAILABLE:
        mlflow.end_run()


def log_metric(*args, **kwargs):
    if MLFLOW_AVAILABLE:
        mlflow.log_metric(*args, **kwargs)
