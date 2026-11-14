"""Model families behind one interface: fit(train_df, feature_cols) / predict(df) -> DataFrame[TARGETS].

* NaiveBaseline  – value from 24 h ago (prediction for T is the observation at T-24 = current obs)
* SarimaxModel   – statistical: SARIMAX (AR(1) errors) with a compact exogenous set per target
* RidgeModel / RFModel – classical ML (multi-output)
* LSTMModel      – deep learning (PyTorch), 24 h window -> all targets
Everything stored inside a bundle is plain numpy/sklearn so unpickling needs neither torch nor statsmodels."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config as C

SEQ_COLS = C.BASE_COLS + ["hour_sin", "hour_cos", "month_sin", "month_cos"]


class NaiveBaseline:
    name = "naive"

    def fit(self, df, feature_cols):
        return self

    def predict(self, df):
        return pd.DataFrame({t: df[b].values for t, b in C.TARGET_BASE.items()}, index=df.index)


class RidgeModel:
    name = "ridge"

    def fit(self, df, feature_cols):
        self.cols = feature_cols
        self.pipe = make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(df[feature_cols], df[C.TARGETS])
        return self

    def predict(self, df):
        return pd.DataFrame(self.pipe.predict(df[self.cols]), index=df.index, columns=C.TARGETS)


class RFModel:
    name = "rf"

    def fit(self, df, feature_cols):
        self.cols = feature_cols
        self.model = RandomForestRegressor(n_estimators=120, max_depth=14, min_samples_leaf=5, max_features=0.4,
                                           n_jobs=-1, random_state=C.SEED).fit(df[feature_cols], df[C.TARGETS])
        return self

    def predict(self, df):
        return pd.DataFrame(self.model.predict(df[self.cols]), index=df.index, columns=C.TARGETS)


class SarimaxModel:
    """Regression with AR(1) errors, fit by SARIMAX per target. Exogenous set: the target's own current
    value, 24 h mean/lag and calendar harmonics (compact to keep the state-space fit fast).
    At prediction time the AR error carries no information about an unseen future, so prediction is
    intercept + X·beta; the AR term only improves the coefficient estimates (GLS-like)."""
    name = "sarimax"

    def _exog_cols(self, base):
        return [c for c in [base, f"{base}_roll_mean_24h", f"{base}_lag_24h", f"{base}_diff_3h",
                            "hour_sin", "hour_cos", "month_sin", "month_cos"]]

    def fit(self, df, feature_cols):
        from statsmodels.tsa.statespace.sarimax import SARIMAX  # lazy
        import warnings
        self.fits = {}
        for t, base in C.TARGET_BASE.items():
            cols = self._exog_cols(base)
            X = df[cols].astype(float)
            mu, sd = X.mean(), X.std().replace(0, 1.0)
            Xs = (X - mu) / sd
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = SARIMAX(df[t].astype(float).values, exog=Xs.values, order=(1, 0, 0), trend="c").fit(disp=False, maxiter=60)
                p = np.asarray(res.params)
                intercept, beta = p[0], p[1:1 + len(cols)]
            except Exception:       # non-convergence -> closed-form OLS so the family never breaks the run
                A = np.c_[np.ones(len(Xs)), Xs.values]
                coef = np.linalg.lstsq(A, df[t].astype(float).values, rcond=None)[0]
                intercept, beta = coef[0], coef[1:]
            self.fits[t] = (cols, mu.values, sd.values, float(intercept), np.asarray(beta))
        return self

    def predict(self, df):
        out = {}
        for t, (cols, mu, sd, b0, beta) in self.fits.items():
            out[t] = b0 + ((df[cols].astype(float).values - mu) / sd) @ beta
        return pd.DataFrame(out, index=df.index)


class LSTMModel:
    name = "lstm"

    def __init__(self, epochs: int | None = None, hidden: int = 64):
        self.epochs, self.hidden = epochs or C.LSTM_EPOCHS, hidden
        self.state = None   # numpy arrays only

    # -- sequences
    def _windows(self, X: np.ndarray):
        W = C.LSTM_WINDOW
        idx = np.arange(W - 1, len(X))
        return np.stack([X[i - W + 1:i + 1] for i in idx]), idx

    def _net(self, n_in, n_out):
        import torch.nn as nn

        class Net(nn.Module):
            def __init__(s):
                super().__init__()
                s.lstm = nn.LSTM(n_in, self.hidden, batch_first=True)
                s.head = nn.Sequential(nn.Linear(self.hidden, 64), nn.ReLU(), nn.Linear(64, n_out))

            def forward(s, x):
                o, _ = s.lstm(x)
                return s.head(o[:, -1])
        return Net()

    def fit(self, df, feature_cols):
        import torch
        torch.manual_seed(C.SEED)
        torch.set_num_threads(2)
        X = df[SEQ_COLS].astype(float).values
        Y = df[C.TARGETS].astype(float).values
        self.xm, self.xs = X.mean(0), np.where(X.std(0) == 0, 1, X.std(0))
        self.ym, self.ys = Y.mean(0), np.where(Y.std(0) == 0, 1, Y.std(0))
        Xw, idx = self._windows((X - self.xm) / self.xs)
        Yw = ((Y - self.ym) / self.ys)[idx]
        n_val = max(int(len(Xw) * 0.1), 1)
        tr_x, tr_y = torch.tensor(Xw[:-n_val], dtype=torch.float32), torch.tensor(Yw[:-n_val], dtype=torch.float32)
        va_x, va_y = torch.tensor(Xw[-n_val:], dtype=torch.float32), torch.tensor(Yw[-n_val:], dtype=torch.float32)
        net = self._net(X.shape[1], Y.shape[1])
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        loss_fn = torch.nn.MSELoss()
        best, best_state, patience = 1e9, None, 0
        for ep in range(self.epochs):
            net.train()
            perm = torch.randperm(len(tr_x))
            for i in range(0, len(perm), 64):
                b = perm[i:i + 64]
                opt.zero_grad()
                loss_fn(net(tr_x[b]), tr_y[b]).backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                v = loss_fn(net(va_x), va_y).item()
            if v < best - 1e-4:
                best, patience = v, 0
                best_state = {k: t.detach().cpu().numpy().copy() for k, t in net.state_dict().items()}
            else:
                patience += 1
                if patience >= 5:
                    break
        self.state = best_state
        self.n_in, self.n_out = X.shape[1], Y.shape[1]
        return self

    def predict(self, df):
        import torch
        net = self._net(self.n_in, self.n_out)
        net.load_state_dict({k: torch.tensor(v) for k, v in self.state.items()})
        net.eval()
        X = (df[SEQ_COLS].astype(float).values - self.xm) / self.xs
        out = pd.DataFrame(np.nan, index=df.index, columns=C.TARGETS)
        if len(X) < C.LSTM_WINDOW:
            return out
        Xw, idx = self._windows(X)
        with torch.no_grad():
            p = net(torch.tensor(Xw, dtype=torch.float32)).numpy() * self.ys + self.ym
        out.iloc[idx] = p
        return out


FAMILIES = {"sarimax": SarimaxModel, "ridge": RidgeModel, "rf": RFModel, "lstm": LSTMModel}
FAMILY_TYPE = {"naive": "baseline", "sarimax": "statistical", "ridge": "classical ML",
               "rf": "classical ML", "lstm": "deep learning"}


# ---------------------------------------------------------------- post-processing

def postprocess(pred: pd.DataFrame) -> pd.DataFrame:
    """Raw target-space predictions -> user-facing forecast columns."""
    o = pd.DataFrame(index=pred.index)
    g = lambda c: pred[f"target_{c}"]
    o["aqi"] = g("aqi").clip(0, 500)
    o["temperature"] = g("temperature")
    o["humidity"] = g("humidity").clip(0, 100)
    o["uv_index"] = g("uv_index").clip(0, 16)
    o["precip_mm"] = g("precip_mm").clip(lower=0)
    o["rain_prob"] = g("rain_flag").clip(0, 1)
    o["wind_speed"] = g("wind_speed").clip(lower=0)
    o["wind_dir"] = (np.degrees(np.arctan2(g("wind_dir_sin"), g("wind_dir_cos"))) % 360)
    return o


@dataclass
class Bundle:
    """What gets stored in the Model Registry."""
    models: dict
    choice: dict                 # target -> family
    feature_cols: list
    val_rmse: dict               # target -> rmse of the chosen model on validation
    val_std: dict
    val_wind_dir_mae: float
    score: float                 # mean(rmse/std) – lower is better
    train_start: str = ""
    train_end: str = ""
    trained_at: str = ""
    extra: dict = field(default_factory=dict)

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        cache = {}
        out = pd.DataFrame(index=df.index, columns=C.TARGETS, dtype=float)
        for t, fam in self.choice.items():
            if fam not in cache:
                cache[fam] = self.models[fam].predict(df)
            out[t] = cache[fam][t]
        return out
