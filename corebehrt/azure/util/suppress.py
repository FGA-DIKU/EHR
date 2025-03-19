import sys
import contextlib

SUPRESS_WARNINGS = [
    "pathOnCompute is not a known attribute of class",
    "This is an experimental class",
    "Found the config file in: /config.json",
]


@contextlib.contextmanager
def suppress_warnings():
    backup = sys.stderr
    sys.stderr = AzureWarningFilter(sys.stderr)
    yield
    sys.stderr = backup


class AzureWarningFilter(object):
    def __init__(self, stream):
        self.stream = stream

    def __getattr__(self, attr_name):
        return getattr(self.stream, attr_name)

    def write(self, data):
        for pattern in SUPRESS_WARNINGS:
            if pattern in data:
                return
        self.stream.write(data)
