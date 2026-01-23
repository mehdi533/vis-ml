from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_utils_local import SCALER_FACTORY, RECOMMENDED_SCALER_FACTORY


SCALER_CATALOG = sorted(SCALER_FACTORY.keys())
RECOMMENDED_SCALER_CATALOG = sorted(RECOMMENDED_SCALER_FACTORY.keys())


def list_scalers():
    return list(SCALER_CATALOG)


def list_recommended_scalers():
    return list(RECOMMENDED_SCALER_CATALOG)
