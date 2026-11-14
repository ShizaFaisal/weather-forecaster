"""Write EDA tables for every metric to reports/eda/ (the dashboard renders the interactive version)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src import config as C
from src import eda
from src.features import from_store_frame
from src.store import get_store

if __name__ == "__main__":
    d = eda.prepare(from_store_frame(get_store().read(C.FG_FEATURES)))
    out = C.ROOT / "reports" / "eda"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({c: eda.profile(d, c) for c in eda.EDA_COLS}).T.to_csv(out / "summary.csv")
    eda.correlation(d).to_csv(out / "correlation.csv")
    for c in eda.EDA_COLS:
        eda.hourly_profile(d, c).to_csv(out / f"hourly_{c}.csv")
    print("wrote", out)
