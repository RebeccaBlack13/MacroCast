"""
Shared framework for the macro/market forecasting models.

Conventions (applies to every model)
------------------------------------
* Forecast origin = end of month t.
* Macro series (CPI, core CPI, IP, M3) are assumed published with a 1-month lag,
  so features at origin t only use macro data through month t-1.
* VIX features use data through the end of month t.
* Regression targets are CHANGES vs the last observed value (e.g. YoY(t+h) - YoY(t-1)),
  so the "no-change / persistence" benchmark is simply a forecast of zero.
* Walk-forward (expanding window) evaluation: each model is re-fit periodically and only
  trained on rows whose target was already published at the re-fit date.
"""
import pickle, warnings
import numpy as np, pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import (HistGradientBoostingRegressor, RandomForestRegressor,
                              HistGradientBoostingClassifier, RandomForestClassifier)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss

warnings.filterwarnings("ignore")
D = pickle.load(open("data.pkl", "rb"))
MIDX = pd.date_range("1956-01-01", "2024-05-01", freq="MS")       # last full VIX month = May-2024
PUB_LAG = 1                                                       # months


# ---------------------------------------------------------------- helpers
def mlog(df):
    return np.log(df.reindex(MIDX))


def vix_monthly(assets=("SP500",)):
    v = np.log(D["vix"][list(assets)])
    g = v.groupby(pd.Grouper(freq="MS"))
    parts = {"mean": g.mean(), "last": g.last(), "max": g.max(), "min": g.min(), "std": g.std()}
    out = {}
    for stat, df in parts.items():
        for a in assets:
            out[f"vix_{a.lower()}_{stat}"] = df[a]
    out = pd.DataFrame(out).reindex(MIDX)
    for a in assets:
        m = out[f"vix_{a.lower()}_mean"]
        out[f"vix_{a.lower()}_chg1"] = m.diff(1)
        out[f"vix_{a.lower()}_chg3"] = m.diff(3)
        out[f"vix_{a.lower()}_dev12"] = m - m.rolling(12, min_periods=6).mean()
    return out


def growth_block(lg, name):
    """Standard growth features from a monthly log-index Series (obs-month based)."""
    f = pd.DataFrame(index=lg.index)
    f[f"{name}_mom"] = lg.diff(1)
    f[f"{name}_mom_l1"] = lg.diff(1).shift(1)
    f[f"{name}_g3"] = lg.diff(3)
    f[f"{name}_g6"] = lg.diff(6)
    f[f"{name}_yoy"] = lg.diff(12)
    f[f"{name}_acc"] = lg.diff(6) - lg.shift(6).diff(6)
    return f


def us_obs_features():
    """US macro features indexed by OBSERVATION month (not yet lag-shifted)."""
    cpi = mlog(D["cpi"][["USA"]])["USA"]
    core = mlog(D["core"][["USA"]])["USA"]
    m3 = mlog(D["m3sa"][["USA"]])["USA"]
    ipall = mlog(D["ip"])
    ip, mfg = ipall["Industrial_Production"], ipall["Manufacturing"]
    sectors = ipall.drop(columns=["Industrial_Production", "Manufacturing"])
    F = pd.concat([growth_block(cpi, "cpi"), growth_block(core, "core"), growth_block(m3, "m3"),
                   growth_block(ip, "ip"), growth_block(mfg, "mfg")], axis=1)
    F["gap_cpi_core"] = F["cpi_yoy"] - F["core_yoy"]
    F["real_m3"] = F["m3_yoy"] - F["cpi_yoy"]
    F["real_m3_g6"] = F["m3_g6"] - F["cpi_g6"]
    F["ip_mfg_gap"] = F["ip_yoy"] - F["mfg_yoy"]
    F["breadth3"] = (sectors.diff(3) > 0).where(sectors.diff(3).notna()).mean(axis=1)
    F["cpi_yoy_lr"] = F["cpi_yoy"].rolling(60, min_periods=36).mean()
    F["ip_yoy_lr"] = F["ip_yoy"].rolling(60, min_periods=36).mean()
    F["m3_yoy_lr"] = F["m3_yoy"].rolling(60, min_periods=36).mean()
    return F


TARGET_SERIES = {  # name -> obs-month yoy series builder
    "cpi": lambda: mlog(D["cpi"][["USA"]])["USA"].diff(12),
    "core": lambda: mlog(D["core"][["USA"]])["USA"].diff(12),
    "ip": lambda: mlog(D["ip"][["Industrial_Production"]])["Industrial_Production"].diff(12),
    "m3": lambda: mlog(D["m3sa"][["USA"]])["USA"].diff(12),
}


def us_dataset(target, h, vix=False):
    """Return X (origin-indexed), y (delta vs last observed), base (last observed level)."""
    F = us_obs_features().shift(PUB_LAG)                    # macro known at origin t = obs t-1
    if vix:
        F = F.join(vix_monthly(("SP500",)))
    yoy = TARGET_SERIES[target]()
    base = yoy.shift(PUB_LAG)                               # last observed YoY at origin
    y = yoy.shift(-h) - base
    X = F.copy()
    X.index.name = "date"
    return X, y.rename("y"), base.rename("base")


# ---------------------------------------------------------------- models
class RidgeTS:
    """Ridge with median-imputation + scaling; alpha picked on a chronological hold-out."""
    def __init__(self, alphas=(1, 10, 100, 1000, 10000), cols=None):
        self.alphas, self.cols = alphas, cols

    def _pipe(self, a):
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=a))

    def fit(self, X, y, gap=2):
        Xc = X[self.cols] if self.cols else X
        Xc = Xc.loc[:, Xc.notna().any()]
        self.used = list(Xc.columns)
        n = len(Xc); k = max(int(n * 0.2), 12)
        if n - k - gap > 40:
            tr, va = slice(0, n - k - gap), slice(n - k, n)
            best, ba = np.inf, self.alphas[len(self.alphas) // 2]
            for a in self.alphas:
                p = self._pipe(a).fit(Xc.iloc[tr], y.iloc[tr])
                e = np.mean((p.predict(Xc.iloc[va]) - y.iloc[va]) ** 2)
                if e < best: best, ba = e, a
        else:
            ba = self.alphas[len(self.alphas) // 2]
        self.alpha_ = ba
        self.m = self._pipe(ba).fit(Xc, y)
        return self

    def predict(self, X):
        return self.m.predict(X[self.used])


class SK:
    """Thin wrapper so tree models share the fit/predict API (and drop all-NaN columns)."""
    def __init__(self, make, impute=False):
        self.make, self.impute = make, impute

    def fit(self, X, y, gap=2):
        Xc = X.loc[:, X.notna().any()]
        self.used = list(Xc.columns)
        self.m = self.make()
        self.make = None                     # drop the factory lambda so fitted models can be pickled
        if self.impute:
            self.imp = SimpleImputer(strategy="median").fit(Xc)
            self.m.fit(self.imp.transform(Xc), y)
        else:
            self.m.fit(Xc, y)
        return self

    def predict(self, X):
        Z = X[self.used]
        return self.m.predict(self.imp.transform(Z) if self.impute else Z)


def hgb(small=True):
    kw = dict(learning_rate=0.04, max_iter=150 if small else 200, max_depth=3,
              min_samples_leaf=15 if small else 60, l2_regularization=1.0, random_state=0)
    return lambda: HistGradientBoostingRegressor(**kw)


def rf(small=True):
    if small:
        kw = dict(n_estimators=200, max_depth=5, min_samples_leaf=8, max_features=0.4, n_jobs=1, random_state=0)
    else:
        kw = dict(n_estimators=80, max_depth=9, min_samples_leaf=40, max_features=0.33, max_samples=0.5,
                  n_jobs=1, random_state=0)
    return lambda: RandomForestRegressor(**kw)


def model_zoo(own_cols=None, small=True, with_rf=True):
    z = {"Ridge": lambda: RidgeTS(), "HistGB": lambda: SK(hgb(small))}
    if with_rf:
        z["RandForest"] = lambda: SK(rf(small), impute=True)
    if own_cols:
        z["AR(own lags)"] = lambda: RidgeTS(cols=own_cols)
    return z


# ---------------------------------------------------------------- walk-forward
def walk_forward(X, y, models, test_start, refit_months=12, h=1, train_start=None,
                 dates=None, clip=None, min_train=60, end=None, verbose=False, pub_lag=None):
    """
    Expanding-window walk-forward.  X,y share an index (panel: duplicated dates allowed, pass `dates`).
    Rows are usable for training at re-fit date R only if date <= R - (h+PUB_LAG) months.
    Returns DataFrame with y, and one column per model plus 'Ensemble'.
    """
    lag = PUB_LAG if pub_lag is None else pub_lag          # months between observation and publication
    dates = pd.DatetimeIndex(dates if dates is not None else X.index)
    ok = y.notna().values
    if train_start is not None: ok &= (dates >= pd.Timestamp(train_start))
    refits = pd.date_range(pd.Timestamp(test_start), (pd.Timestamp(end) if end else dates.max()) + pd.offsets.MonthBegin(1),
                           freq=f"{refit_months}MS")
    preds = []
    for i, R in enumerate(refits):
        R2 = refits[i + 1] if i + 1 < len(refits) else dates.max() + pd.offsets.MonthBegin(1)
        tr = ok & (dates <= R - pd.DateOffset(months=h + lag))
        te = (dates >= R) & (dates < R2) & y.notna().values
        if tr.sum() < min_train or te.sum() == 0: continue
        Xtr, ytr = X[tr], y[tr]
        if clip is not None: ytr = ytr.clip(-clip, clip)
        out = pd.DataFrame({"y": y[te].values}, index=X.index[te])
        out["date"] = dates[te]
        for name, mk in models.items():
            m = mk().fit(Xtr, ytr, gap=h + lag)
            out[name] = m.predict(X[te])
        preds.append(out)
        if verbose: print("  refit", R.date(), tr.sum(), te.sum(), flush=True)
    P = pd.concat(preds)
    mcols = [c for c in models]
    ens = [c for c in mcols if not c.startswith("AR")]
    if len(ens) > 1: P["Ensemble"] = P[ens].mean(axis=1)
    P.insert(1, "Naive(0)", 0.0)
    return P


# ---------------------------------------------------------------- metrics
def dm_test(d, h):
    """Diebold-Mariano (HLN small-sample) on loss differential d = loss_naive - loss_model (>0 = model better)."""
    d = np.asarray(d, float); d = d[~np.isnan(d)]; n = len(d)
    if n < 20: return np.nan, np.nan
    db = d.mean(); L = max(h, 1); s = np.mean((d - db) ** 2)
    for k in range(1, L + 1):
        s += 2 * (1 - k / (L + 1)) * np.mean((d[k:] - db) * (d[:-k] - db))
    s = max(s, 1e-14)
    st = db / np.sqrt(s / n) * np.sqrt(max((n + 1 - 2 * h + h * (h - 1) / n) / n, 1e-6))
    return st, 1 - stats.t.cdf(st, n - 1)                         # one-sided: model better than naive


def evaluate(P, h, label="", by=None):
    """Score a prediction frame.  Returns tidy rows: model, RMSE, MAE, rel_RMSE_vs_naive, DM p-value."""
    rows = []
    e_naive = (P["y"] - P["Naive(0)"]) ** 2
    for c in [c for c in P.columns if c not in ("y", "date") and pd.api.types.is_numeric_dtype(P[c])]:
        e = (P["y"] - P[c]) ** 2
        st, p = (np.nan, np.nan) if c == "Naive(0)" else dm_test((e_naive - e).values, h)
        rows.append(dict(target=label, model=c, n=len(P),
                         RMSE=float(np.sqrt(e.mean())), MAE=float((P["y"] - P[c]).abs().mean()),
                         rel_RMSE=float(np.sqrt(e.mean() / e_naive.mean())), DM_p_onesided=p))
    return rows
