"""Daily training pipeline (also triggered by the Drift Agent)."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import logs
from src.store import get_store
from src.training import train

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="promote even if not better than the current model")
    ap.add_argument("--families", nargs="*", help="subset of: sarimax ridge rf lstm")
    a = ap.parse_args()
    store = get_store()
    try:
        print(train(store, a.families, promote_always=a.force))
    finally:
        logs.flush(store)
