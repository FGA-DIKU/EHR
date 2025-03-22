from corebehrt.azure.util.suppress import suppress_warnings

AZURE_AVAILABLE = False

try:
    #
    # Check if azure is available and set flag.
    #
    with suppress_warnings():
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
