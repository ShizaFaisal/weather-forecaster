"""SHAP (stored in the Feature Store at training time) and LIME (on demand in the dashboard)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from . import logs


def explain_rf(store, rf_model, val_df: pd.DataFrame, fcols: list[str], run_id: str, n: int = 150, top: int = 15):
    import shap
    X = val_df[fcols].sample(min(n, len(val_df)), random_state=C.SEED)
    sv = shap.TreeExplainer(rf_model.model).shap_values(X)
    sv = np.stack(sv, axis=-1) if isinstance(sv, list) else np.asarray(sv)   # (rows, features, outputs)
    rows = []
    for j, t in enumerate(C.TARGETS):
        imp = np.abs(sv[:, :, j]).mean(axis=0)
        for k in np.argsort(imp)[::-1][:top]:
            rows.append(dict(id=f"{run_id}-{t}-{fcols[k]}", ts=logs.now_utc(), run_id=run_id, target=t,
                             feature=fcols[k], mean_abs_shap=float(imp[k]), method="shap_tree_rf"))
    store.write(C.FG_EXPLAIN, pd.DataFrame(rows))


def lime_explain(rf_model, train_X: pd.DataFrame, row: pd.Series, target: str, num_features: int = 10):
    """LIME explanation of the RF prediction for one feature row -> DataFrame(feature, weight)."""
    from lime.lime_tabular import LimeTabularExplainer
    j = C.TARGETS.index(target)
    cols = list(train_X.columns)
    exp = LimeTabularExplainer(train_X.values, feature_names=cols, mode="regression", random_state=C.SEED)
    e = exp.explain_instance(row[cols].values.astype(float), lambda a: rf_model.model.predict(pd.DataFrame(a, columns=cols))[:, j],
                             num_features=num_features, num_samples=800)
    return pd.DataFrame(e.as_list(), columns=["feature", "weight"])
