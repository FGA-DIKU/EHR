import sys

AZURE_AVAILABLE = False

try:
    #
    # Check if azure is available and set flag.
    #
    from azure.ai.ml import MLClient
    from azure.identity import DefaultAzureCredential

    AZURE_AVAILABLE = True
except:
    pass


def is_azure_available() -> bool:
    """
    Checks if Azure modules are available.

    :return: True if available, otherwise False.
    """
    global AZURE_AVAILABLE
    return AZURE_AVAILABLE


def check_azure() -> None:
    """
    Checks if Azure modules are available, raises an exception if not.
    """
    if not is_azure_available():
        raise Exception("Azure modules not found!")


def ml_client() -> "MLClient":
    """
    Returns the Azure MLClient.
    """
    check_azure()
    return MLClient.from_config(DefaultAzureCredential())


class AzureWarningSupressor:
    def __init__(self, patterns):
        self.patterns = patterns

    def __enter__(self):
        sys.stdout = AzurePrintFilter(sys.stdout, self.patterns)

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = sys.stdout.stream


class AzurePrintFilter(object):
    def __init__(self, stream, patterns):
        self.stream = stream
        self.patterns = patterns
        self.triggered = False

    def __getattr__(self, attr_name):
        return getattr(self.stream, attr_name)

    def write(self, data):
        if data == "\n" and self.triggered:
            self.triggered = False
        else:
            for pattern in self.patterns:
                if pattern in data:
                    self.triggered = True
                    return
            self.stream.write(data)
            self.stream.flush()

    def flush(self):
        self.stream.flush()
