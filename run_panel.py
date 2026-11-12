"""Panel models + classifiers.

  A. Monthly CPI panel (44 countries, pooled)   : YoY(t+h) - YoY(t-1),  h = 3, 6, 12
  B. Quarterly CPI panel (47 economies incl AUS, NZL, which are quarterly-only): h = 1, 2, 4 quarters
  C. US industrial-production sector panel (21 series): h = 3, 6
  D. Classifiers (US): IP YoY < 0 in 6 months;  CPI YoY > 3% in 12 months
All walk-forward, expanding window, publication lag respected.
"""
import pickle, time
import numpy as np, pandas as pd
from ml import *
from run_vix import Clf

t0 = time.time()
GLOBAL_VIX = vix_monthly(("SP500",))[["vix_sp500_mean", "vix_sp500_chg3"]]


def own_cols(name="cpi"):
    return [f"{name}_{s}" for s in ("mom", "mom_l1", "g3", "g6", "yoy", "acc", "yoy_lr")]


# ======================================================== A. monthly CPI panel
def build_cpi_panel(h):
    cpi = mlog(D["cpi"])
    yoy_all = cpi.diff(12)
    glob = yoy_all.drop(columns=["TUR"]).median(axis=1)            # robust global inflation
    core = mlog(D["core"])
    m3 = mlog(D["m3sa"])
    m3n = mlog(D["m3nsa"])
    us_yoy = yoy_all["USA"]
    frames = []
    for c in cpi.columns:
        lg = cpi[c]
        F = growth_block(lg, "cpi")
        F["cpi_yoy_lr"] = F["cpi_yoy"].rolling(60, min_periods=36).mean()
        F["core_yoy"] = core[c].diff(12) if c in core else np.nan
        F["core_g3"] = core[c].diff(3) if c in core else np.nan
        mm = m3[c] if c in m3 else (m3n[c] if c in m3n else None)
        F["m3_yoy"] = mm.diff(12) if mm is not None else np.nan
        F["m3_g6"] = mm.diff(6) if mm is not None else np.nan
        F["glob_yoy"] = glob
        F["rel_glob"] = F["cpi_yoy"] - glob
        F["us_yoy"] = us_yoy
        F = F.shift(PUB_LAG).join(GLOBAL_VIX)
        F["country"] = c
        F["y"] = yoy_all[c].shift(-h) - yoy_all[c].shift(PUB_LAG)
        F["base"] = yoy_all[c].shift(PUB_LAG)
        F["date"] = F.index
        frames.append(F.dropna(subset=["cpi_yoy"]))
    L = pd.concat(frames, ignore_index=True)
    return L


def run_cpi_panel():
    rows, preds, pc_rows = [], {}, []
    for h in (3, 6, 12):
        L = build_cpi_panel(h)
        L = L[L["date"] >= "1962-01-01"].sort_values("date", kind="stable").reset_index(drop=True)
        meta = L[["date", "country", "base"]]
        X = L.drop(columns=["country", "y", "base", "date"])
        own = own_cols("cpi")
        P = walk_forward(X, L["y"], model_zoo(own, small=False, with_rf=False), test_start="2005-01-01", h=h,
                         refit_months=12, dates=L["date"], clip=0.10, min_train=2000)
        P = P.join(meta[["country"]])
        preds[f"cpi_panel_h{h}"] = P
        for label, sub in (("all", P), ("ex_TUR", P[P.country != "TUR"])):
            rows += evaluate(sub, h, f"cpi_panel|h{h}|{label}")
        # per-country relative RMSE for best generic model (Ensemble)
        for c, g in P.groupby("country"):
            if len(g) < 60: continue
            e0 = ((g["y"] - g["Naive(0)"]) ** 2).mean()
            pc_rows.append(dict(h=h, country=c, n=len(g),
                                rel_RMSE_Ensemble=float(np.sqrt(((g["y"] - g["Ensemble"]) ** 2).mean() / e0)),
                                rel_RMSE_AR=float(np.sqrt(((g["y"] - g["AR(own lags)"]) ** 2).mean() / e0))))
        print(f"cpi panel h={h} done ({time.time()-t0:.0f}s)", flush=True)
    return pd.DataFrame(rows), preds, pd.DataFrame(pc_rows)


# ======================================================== B. quarterly CPI panel (adds AUS, NZL)
def _quarterly_base():
    cq = D["cpiq"]
    QI = pd.date_range("1955-01-01", "2024-04-01", freq="QS")   # last row = forecast origin (y is NaN)
    lg = np.log(cq.reindex(QI))
    yoy = lg.diff(4)
    glob = yoy.drop(columns=["TUR"]).median(axis=1)
    m3 = np.log(D["m3sa"]).copy()
    m3n = np.log(D["m3nsa"]).copy()
    vixq = GLOBAL_VIX.resample("QS").last().reindex(QI)
    return QI, lg, yoy, glob, m3, m3n, vixq


def build_cpiq(hq):
    """Long panel of quarterly CPI features. Rows with y NaN are the latest (forecastable) origins."""
    QI, lg, yoy, glob, m3, m3n, vixq = _quarterly_base()
    frames = []
    for c in lg.columns:
        s = lg[c]
        F = pd.DataFrame(index=QI)
        F["qoq"] = s.diff(1); F["qoq_l1"] = s.diff(1).shift(1); F["g2"] = s.diff(2)
        F["yoy"] = s.diff(4); F["acc"] = s.diff(2) - s.shift(2).diff(2)
        F["yoy_lr"] = F["yoy"].rolling(20, min_periods=12).mean()
        F["glob_yoy"] = glob; F["rel_glob"] = F["yoy"] - glob
        mm = m3[c] if c in m3 else (m3n[c] if c in m3n else None)
        if mm is not None:
            mq = mm.dropna().resample("QS").last().reindex(QI)
            F["m3_yoy"] = mq.diff(4)
        else:
            F["m3_yoy"] = np.nan
        F = F.shift(1).join(vixq)                              # 1-quarter publication lag
        F["country"] = c
        F["y"] = yoy[c].shift(-hq) - yoy[c].shift(1)
        F["base"] = yoy[c].shift(1)
        F["date"] = F.index
        frames.append(F.dropna(subset=["yoy"]))
    L = pd.concat(frames, ignore_index=True)
    return L[L["date"] >= "1962-01-01"].sort_values("date", kind="stable").reset_index(drop=True)


CPIQ_OWN = ["qoq", "qoq_l1", "g2", "yoy", "acc", "yoy_lr"]


def run_cpi_quarterly():
    rows, preds = [], {}
    for hq in (1, 2, 4):
        L = build_cpiq(hq)
        meta = L[["country"]]
        X = L.drop(columns=["country", "y", "date", "base"])
        P = walk_forward(X, L["y"], model_zoo(CPIQ_OWN, small=False, with_rf=False), test_start="2005-01-01",
                         h=3 * hq, refit_months=12, dates=L["date"], clip=0.10, min_train=1500, pub_lag=3)
        P = P.join(meta)
        preds[f"cpi_q_h{hq}"] = P
        for label, sub in (("all", P), ("ex_TUR", P[P.country != "TUR"]), ("AUS", P[P.country == "AUS"]),
                           ("NZL", P[P.country == "NZL"])):
            if len(sub) >= 20:
                rows += evaluate(sub, max(hq, 1), f"cpi_quarterly|h{hq}q|{label}")
        print(f"cpi quarterly h={hq}q done ({time.time()-t0:.0f}s)", flush=True)
    return pd.DataFrame(rows), preds


# ======================================================== C. IP sector panel
IP_OWN = [f"ip_{s}" for s in ("mom", "mom_l1", "g3", "g6", "yoy", "acc", "yoy_lr")]


def build_ip_panel(h):
    """Long panel of IP sector features. Rows with y NaN are the latest (forecastable) origins."""
    ipl = mlog(D["ip"])
    tot = ipl["Industrial_Production"]
    tot_f = growth_block(tot, "tot")
    cpi_yoy = mlog(D["cpi"][["USA"]])["USA"].diff(12)
    m3_yoy = mlog(D["m3sa"][["USA"]])["USA"].diff(12)
    macro = pd.DataFrame({"us_cpi_yoy": cpi_yoy, "us_real_m3": m3_yoy - cpi_yoy})
    frames = []
    for k in ipl.columns:
        s = ipl[k]
        F = growth_block(s, "ip")
        F["ip_yoy_lr"] = F["ip_yoy"].rolling(60, min_periods=36).mean()
        F["rel_tot"] = F["ip_yoy"] - tot_f["tot_yoy"]
        F = pd.concat([F, tot_f[["tot_mom", "tot_g3", "tot_g6", "tot_yoy", "tot_acc"]], macro], axis=1)
        F = F.shift(PUB_LAG).join(GLOBAL_VIX)
        F["series"] = k
        yoy = s.diff(12)
        F["y"] = yoy.shift(-h) - yoy.shift(PUB_LAG)
        F["base"] = yoy.shift(PUB_LAG)
        F["date"] = F.index
        frames.append(F.dropna(subset=["ip_yoy"]))
    L = pd.concat(frames, ignore_index=True)
    return L[L["date"] >= "1976-01-01"].sort_values("date", kind="stable").reset_index(drop=True)


def ip_design(L):
    X = pd.concat([L.drop(columns=["series", "y", "date", "base"]), pd.get_dummies(L["series"], dtype=float)], axis=1)
    return X


def run_ip_sectors():
    rows, preds = [], {}
    for h in (3, 6):
        L = build_ip_panel(h)
        meta = L[["series"]]
        X = ip_design(L)
        P = walk_forward(X, L["y"], model_zoo(IP_OWN, small=False, with_rf=False), test_start="2000-01-01", h=h,
                         refit_months=12, dates=L["date"], clip=0.25, min_train=2000)
        P = P.join(meta)
        preds[f"ip_sector_h{h}"] = P
        rows += evaluate(P, h, f"ip_sectors|h{h}|all")
        covid = (P["date"] >= "2020-03-01") & (P["date"] <= "2021-12-01")
        rows += evaluate(P[~covid], h, f"ip_sectors|h{h}|ex_covid")
        print(f"ip sectors h={h} done ({time.time()-t0:.0f}s)", flush=True)
    return pd.DataFrame(rows), preds


# ======================================================== D. monthly classifiers (US)
def wf_clf_monthly(X, ev, h, test_start="2000-01-01", refit_months=12, min_train=120):
    dates = X.index
    refits = pd.date_range(test_start, dates.max() + pd.offsets.MonthBegin(1), freq=f"{refit_months}MS")
    out = []
    for i, R in enumerate(refits):
        R2 = refits[i + 1] if i + 1 < len(refits) else dates.max() + pd.offsets.MonthBegin(1)
        tr = (dates <= R - pd.DateOffset(months=h + PUB_LAG)) & ev.notna().values
        te = (dates >= R) & (dates < R2) & ev.notna().values
        if tr.sum() < min_train or te.sum() == 0: continue
        o = pd.DataFrame({"y": ev[te].values, "date": dates[te], "BaseRate": float(ev[tr].mean())}, index=dates[te])
        for name, kind in (("Logit", "Logit"), ("HistGB", "HGB")):
            o[name] = Clf(kind).fit(X[tr], ev[tr]).predict(X[te])
        out.append(o)
    P = pd.concat(out)
    P["Ensemble"] = P[["Logit", "HistGB"]].mean(axis=1)
    return P


def run_classifiers():
    rows, preds = [], {}
    # D1: IP YoY < 0 six months ahead
    X, _, _ = us_dataset("ip", 6, vix=True)
    X = X.loc["1962":]
    ev = (TARGET_SERIES["ip"]().shift(-6) < 0).astype(float).where(TARGET_SERIES["ip"]().shift(-6).notna()).reindex(X.index)
    # D2: CPI YoY > 3% twelve months ahead
    X2, _, _ = us_dataset("cpi", 12, vix=True)
    X2 = X2.loc["1962":]
    s = TARGET_SERIES["cpi"]().shift(-12)
    ev2 = (s > 0.03).astype(float).where(s.notna()).reindex(X2.index)
    for name, XX, ee, h in (("US IP YoY < 0 in 6m", X, ev, 6), ("US CPI YoY > 3% in 12m", X2, ev2, 12)):
        P = wf_clf_monthly(XX, ee, h)
        preds[name] = P
        for c in ("BaseRate", "Logit", "HistGB", "Ensemble"):
            rows.append(dict(task=name, model=c, n=len(P), event_rate=P["y"].mean(),
                             AUC=(np.nan if c == "BaseRate" else roc_auc_score(P["y"], P[c])),
                             Brier=brier_score_loss(P["y"], P[c]),
                             Brier_skill=1 - brier_score_loss(P["y"], P[c]) / brier_score_loss(P["y"], P["BaseRate"])))
        print(f"classifier {name} done ({time.time()-t0:.0f}s)", flush=True)
    return pd.DataFrame(rows), preds


if __name__ == "__main__":
    out = {}
    out["cpi_panel_res"], out["cpi_panel_preds"], out["cpi_panel_country"] = run_cpi_panel()
    out["cpiq_res"], out["cpiq_preds"] = run_cpi_quarterly()
    out["ip_res"], out["ip_preds"] = run_ip_sectors()
    out["clf_res"], out["clf_preds"] = run_classifiers()
    pickle.dump(out, open("panel_results.pkl", "wb"))
    print("ALL DONE", round(time.time() - t0), "s")
