"""Daily implied-vol models for the 7 VIX-style indices.

Regression target : log(VIX[t+21 trading days]) - log(VIX[t])      (benchmark = random walk, i.e. 0)
Classifier target : VIX rises >= 30% above today's level at some point in the next 21 trading days
Features          : own level/term-structure-of-history, cross-asset stress, lagged US macro (2-month lag).
Evaluation        : annual re-fit, expanding window, 32-calendar-day embargo between train and test.
The vendor 'Percentile' column is NOT used (it is a full-sample rank and leaks the future).
"""
import pickle, time
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from ml import (D, RidgeTS, SK, hgb, us_obs_features, dm_test, evaluate, roc_auc_score, brier_score_loss,
                HistGradientBoostingClassifier, SimpleImputer, StandardScaler, make_pipeline)

H = 21
TEST_START = "2010-01-01"
EMBARGO = pd.Timedelta(days=32)
LOGV = np.log(D["vix"])
ASSETS = list(LOGV.columns)
MACRO_COLS = ["cpi_yoy", "core_yoy", "ip_yoy", "m3_yoy", "real_m3", "ip_g3", "cpi_acc"]
MAC = us_obs_features()[MACRO_COLS].shift(2)            # known with a ~2-month lag
MAC = MAC.reindex(pd.date_range(MAC.index.min(), "2024-06-01", freq="MS")).ffill(limit=1)   # last (partial) month: carry forward


def feats(asset):
    s = LOGV[asset].dropna()
    f = pd.DataFrame(index=s.index)
    f["lvl"] = s
    for k in (5, 21, 63, 252):
        f[f"dev{k}"] = s - s.rolling(k, min_periods=max(3, k // 2)).mean()
    for k in (1, 5, 21):
        f[f"chg{k}"] = s.diff(k)
    f["vol21"] = s.diff().rolling(21, min_periods=10).std()
    f["rank252"] = s.rolling(252, min_periods=126).rank(pct=True)
    sp = LOGV["SP500"].reindex(s.index).ffill(limit=3)
    f["sp_lvl"], f["sp_chg5"], f["sp_chg21"], f["spread_sp"] = sp, sp.diff(5), sp.diff(21), s - sp
    others = [a for a in ASSETS if a != asset]
    dev = pd.concat([(LOGV[a] - LOGV[a].rolling(21, min_periods=10).mean()) for a in others], axis=1)
    f["xs_dev21"] = dev.reindex(s.index).ffill(limit=3).mean(axis=1)
    m = MAC.reindex(s.index.to_period("M").to_timestamp())
    m.index = s.index
    f = pd.concat([f, m], axis=1)
    y = (s.shift(-H) - s).rename("y")
    fmax = s.rolling(H).max().shift(-H)
    ev = ((fmax - s) >= np.log(1.3)).astype(float).where(fmax.notna()).rename("event")
    return f, y, ev


class Clf:
    def __init__(self, kind):
        self.kind = kind

    def fit(self, X, y):
        Xc = X.loc[:, X.notna().any()]
        self.used = list(Xc.columns)
        if self.kind == "Logit":
            self.m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                   LogisticRegression(C=0.05, max_iter=500))
        else:
            self.m = HistGradientBoostingClassifier(learning_rate=0.04, max_iter=150, max_depth=3,
                                                    min_samples_leaf=60, l2_regularization=2.0, random_state=0)
        self.m.fit(Xc, y.astype(int))
        return self

    def predict(self, X):
        return self.m.predict_proba(X[self.used])[:, 1]


def wf_daily(X, y, models, kind="reg", min_train=750):
    dates = X.index
    refits = pd.date_range(TEST_START, dates.max() + pd.Timedelta(days=366), freq="YS")
    out = []
    for i, R in enumerate(refits):
        R2 = R + pd.offsets.YearBegin(1)
        tr = (dates <= R - EMBARGO) & y.notna().values
        te = (dates >= R) & (dates < R2) & y.notna().values
        if tr.sum() < min_train or te.sum() == 0:
            continue
        o = pd.DataFrame({"y": y[te].values, "date": dates[te]}, index=dates[te])
        if kind == "clf":
            o["BaseRate"] = float(y[tr].mean())
        for name, mk in models.items():
            m = mk().fit(X[tr], y[tr], gap=H) if kind == "reg" else mk().fit(X[tr], y[tr])
            o[name] = m.predict(X[te])
        out.append(o)
    return pd.concat(out)


if __name__ == "__main__":
    t0 = time.time()
    reg_rows, clf_rows, preds, clf_preds = [], [], {}, {}
    for a in ASSETS:
        f, y, ev = feats(a)
        own = ["lvl", "dev5", "dev21", "dev63", "dev252", "chg1", "chg5", "chg21", "vol21", "rank252"]
        models = {"Ridge": lambda: RidgeTS(), "HistGB": lambda: SK(hgb(False)),
                  "AR(own lags)": lambda: RidgeTS(cols=own)}
        P = wf_daily(f, y, models)
        P["Ensemble"] = P[["Ridge", "HistGB"]].mean(axis=1)
        P.insert(2, "Naive(0)", 0.0)
        preds[a] = P
        rows = evaluate(P, H, a)
        reg_rows += rows
        r = {x["model"]: x["rel_RMSE"] for x in rows}
        print(f"{a:12s} n={len(P):5d} {P['date'].min().date()}..{P['date'].max().date()} "
              f"relRMSE ridge={r['Ridge']:.3f} hgb={r['HistGB']:.3f} ar={r['AR(own lags)']:.3f} ens={r['Ensemble']:.3f}"
              f"  ({time.time()-t0:.0f}s)", flush=True)

        C = wf_daily(f, ev, {"Logit": lambda: Clf("Logit"), "HistGB": lambda: Clf("HGB")}, kind="clf")
        C["Ensemble"] = C[["Logit", "HistGB"]].mean(axis=1)
        clf_preds[a] = C
        for c in ("BaseRate", "Logit", "HistGB", "Ensemble"):
            clf_rows.append(dict(asset=a, model=c, n=len(C), event_rate=C["y"].mean(),
                                 AUC=(np.nan if c == "BaseRate" else roc_auc_score(C["y"], C[c])),
                                 Brier=brier_score_loss(C["y"], C[c]),
                                 Brier_skill=1 - brier_score_loss(C["y"], C[c]) / brier_score_loss(C["y"], C["BaseRate"])))
    pickle.dump(dict(reg=pd.DataFrame(reg_rows), clf=pd.DataFrame(clf_rows), preds=preds, clf_preds=clf_preds),
                open("vix_results.pkl", "wb"))
    print("ALL DONE", round(time.time() - t0), "s")
