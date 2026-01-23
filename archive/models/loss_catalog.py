from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from losses_local import LOSS_FACTORY


LOSS_CATALOG = sorted(LOSS_FACTORY.keys())


def list_losses():
    return list(LOSS_CATALOG)
