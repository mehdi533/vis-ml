from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models_local import MODEL_FACTORY


MODEL_CATALOG = {name: cls.__name__ for name, cls in MODEL_FACTORY.items()}


def list_models():
    return sorted(MODEL_CATALOG.keys())
