from torch.utils.data import DataLoader
from tqdm import tqdm


class TqdmToLogger(object):
    """File-like object to redirect tqdm to logger"""

    def __init__(self, logger):
        self.logger = logger

    def write(self, buf):
        # Only log if x is not empty
        for line in buf.rstrip().splitlines():
            x = line.rstrip()
            if x:
                self.logger.info(x)

    def flush(self):
        pass


def get_tqdm(dataloader: DataLoader) -> tqdm:
    return tqdm(dataloader, total=len(dataloader), mininterval=10)
