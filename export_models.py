"""Fit FINAL models on all available data, save them, and produce latest forecasts.

Outputs
  final_models/*.joblib   fitted model objects (dict keyed by (series, horizon, model))
  forecasts.pkl           tidy forecast tables used by build_workbook.py

Forecast conventions are identical to the validation runs: macro data are assumed published with a
1-month lag (3 months for quarterly CPI), regression models predict the CHANGE in YoY vs the last
observed YoY, so  predicted YoY = last observed YoY + predicted change.
NOTE: the uploaded data end in May/June 2024, so these forecasts refer to mid/late 2024.
"""
import os, pickle, joblib
import numpy as np, pandas as pd
from ml import *
from run_panel import (build_cpi_panel, build_cpiq, build_ip_panel, ip_design, own_cols,
                       CPIQ_OWN, IP_OWN, GLOBAL_VIX)
from run_vix import feats as vix_feats, Clf, ASSETS, LOGV

OUT = "final_models"
os.makedirs(OUT, exist_ok=True)


def ens(row, names):
    return float(np.mean([row[n] for n in names]))


# ---------------------------------------------------------------- US monthly
us_rows, us_models = [], {}
for tg in ("cpi", "core", "ip", "m3"):
    for h in (1, 3, 6, 12):
        X, y, base = us_dataset(tg, h, vix=False)
        X, y, base = X.loc["1962":], y.loc["1962":], base.loc["1962":]
        own = [f"{tg}_{s}" for s in ("mom", "mom_l1", "g3", "g6", "yoy", "acc")]
        tr = y.notna().values
        origin = X.index[(X.notna().all(axis=1) & base.notna()).values][-1]   # latest origin with complete features
        xo = X.loc[[origin]]
        r = dict(series=tg, h=h, origin=origin, target_period=origin + pd.DateOffset(months=h),
                 last_obs_yoy=float(base.loc[origin]), feat_missing=float(xo.isna().mean(axis=1).iloc[0]))
        for name, mk in model_zoo(own).items():
            m = mk().fit(X[tr], y[tr], gap=h + PUB_LAG)
            us_models[(tg, h, name)] = m
            r[name] = float(m.predict(xo)[0])
        r["Ensemble"] = ens(r, ["Ridge", "HistGB", "RandForest"])
        r["Naive(0)"] = 0.0
        us_rows.append(r)
joblib.dump(us_models, f"{OUT}/us_monthly_models.joblib", compress=3)
print("US monthly final models:", len(us_models), flush=True)

# ---------------------------------------------------------------- daily VIX
vix_rows, vix_models = [], {}
own_v = ["lvl", "dev5", "dev21", "dev63", "dev252", "chg1", "chg5", "chg21", "vol21", "rank252"]
for a in ASSETS:
    f, y, ev = vix_feats(a)
    tr = y.notna().values
    xo = f.iloc[[-1]]
    last_date = f.index[-1]
    level = float(np.exp(LOGV[a].dropna().iloc[-1]))
    r = dict(asset=a, as_of=last_date, target_date=pd.bdate_range(last_date, periods=22)[-1],
             last_level=level, feat_missing=float(xo.isna().mean(axis=1).iloc[0]))
    for name, mk in {"Ridge": lambda: RidgeTS(), "HistGB": lambda: SK(hgb(False)),
                     "AR(own lags)": lambda: RidgeTS(cols=own_v)}.items():
        m = mk().fit(f[tr], y[tr], gap=21)
        vix_models[(a, 21, name)] = m
        r[name] = float(m.predict(xo)[0])
    r["Ensemble"] = ens(r, ["Ridge", "HistGB"])
    r["Naive(0)"] = 0.0
    vix_rows.append(r)
joblib.dump(vix_models, f"{OUT}/vix_daily_models.joblib", compress=3)
print("VIX final models:", len(vix_models), flush=True)

# ---------------------------------------------------------------- monthly CPI panel
cpi_rows, cpi_models = [], {}
for h in (3, 6, 12):
    L = build_cpi_panel(h)
    L = L[L["date"] >= "1962-01-01"].sort_values("date", kind="stable").reset_index(drop=True)
    X = L.drop(columns=["country", "y", "base", "date"])
    tr = L["y"].notna().values
    yt = L["y"].clip(-0.10, 0.10)
    latest = L.loc[L.groupby("country")["date"].idxmax()]
    latest = latest[latest["date"] >= "2024-03-01"]
    for name, mk in model_zoo(own_cols("cpi"), small=False, with_rf=False).items():
        m = mk().fit(X[tr], yt[tr], gap=h + PUB_LAG)
        cpi_models[(h, name)] = m
        latest[name] = m.predict(X.loc[latest.index])
    cpi_rows.append(latest.assign(h=h, origin=latest["date"])[["country", "h", "origin", "base", "Ridge", "HistGB", "AR(own lags)"]])
cpi_fc = pd.concat(cpi_rows, ignore_index=True)
cpi_fc["Ensemble"] = cpi_fc[["Ridge", "HistGB"]].mean(axis=1)
joblib.dump(cpi_models, f"{OUT}/cpi_panel_models.joblib", compress=3)
print("CPI panel final models:", len(cpi_models), flush=True)

# ---------------------------------------------------------------- quarterly CPI panel (AUS, NZL, ...)
q_rows, q_models = [], {}
for hq in (1, 2, 4):
    L = build_cpiq(hq)
    X = L.drop(columns=["country", "y", "base", "date"])
    tr = L["y"].notna().values
    yt = L["y"].clip(-0.10, 0.10)
    latest = L.loc[L.groupby("country")["date"].idxmax()]
    latest = latest[latest["date"] >= "2024-01-01"]
    for name, mk in model_zoo(CPIQ_OWN, small=False, with_rf=False).items():
        m = mk().fit(X[tr], yt[tr], gap=3 * hq + 3)
        q_models[(hq, name)] = m
        latest[name] = m.predict(X.loc[latest.index])
    q_rows.append(latest.assign(hq=hq, origin=latest["date"])[["country", "hq", "origin", "base", "Ridge", "HistGB", "AR(own lags)"]])
q_fc = pd.concat(q_rows, ignore_index=True)
q_fc["Ensemble"] = q_fc[["Ridge", "HistGB"]].mean(axis=1)
joblib.dump(q_models, f"{OUT}/cpi_quarterly_models.joblib", compress=3)
print("CPI quarterly final models:", len(q_models), flush=True)

# ---------------------------------------------------------------- IP sector panel
ip_rows, ip_models = [], {}
for h in (3, 6):
    L = build_ip_panel(h)
    X = ip_design(L)
    tr = L["y"].notna().values
    yt = L["y"].clip(-0.25, 0.25)
    latest = L.loc[L.groupby("series")["date"].idxmax()]
    for name, mk in model_zoo(IP_OWN, small=False, with_rf=False).items():
        m = mk().fit(X[tr], yt[tr], gap=h + PUB_LAG)
        ip_models[(h, name)] = m
        latest[name] = m.predict(X.loc[latest.index])
    ip_rows.append(latest.assign(h=h, origin=latest["date"])[["series", "h", "origin", "base", "Ridge", "HistGB", "AR(own lags)"]])
ip_fc = pd.concat(ip_rows, ignore_index=True)
ip_fc["Ensemble"] = ip_fc[["Ridge", "HistGB"]].mean(axis=1)
joblib.dump(ip_models, f"{OUT}/ip_sector_models.joblib", compress=3)
print("IP sector final models:", len(ip_models), flush=True)

# ---------------------------------------------------------------- classifiers (US)
clf_rows, clf_models = [], {}
for name, tg, h, event in (("US IP YoY < 0 in 6m", "ip", 6, lambda s: (s < 0.0)),
                           ("US CPI YoY > 3% in 12m", "cpi", 12, lambda s: (s > 0.03))):
    X, _, base = us_dataset(tg, h, vix=True)
    X = X.loc["1962":]
    s = TARGET_SERIES[tg]().shift(-h)
    ev = event(s).astype(float).where(s.notna()).reindex(X.index)
    tr = ev.notna().values
    origin = X.index[(X.notna().all(axis=1) & base.reindex(X.index).notna()).values][-1]
    xo = X.loc[[origin]]
    r = dict(task=name, origin=origin, horizon_months=h, last_obs_yoy=float(base.loc[origin]),
             base_rate=float(ev[tr].mean()))
    for cn, kind in (("Logit", "Logit"), ("HistGB", "HGB")):
        m = Clf(kind).fit(X[tr], ev[tr])
        clf_models[(name, cn)] = m
        r[cn] = float(m.predict(xo)[0])
    r["Ensemble"] = (r["Logit"] + r["HistGB"]) / 2
    clf_rows.append(r)
joblib.dump(clf_models, f"{OUT}/us_classifier_models.joblib", compress=3)

pickle.dump(dict(us=pd.DataFrame(us_rows), vix=pd.DataFrame(vix_rows), cpi=cpi_fc, cpiq=q_fc, ip=ip_fc,
                 clf=pd.DataFrame(clf_rows)), open("forecasts.pkl", "wb"))
print("forecasts saved")
