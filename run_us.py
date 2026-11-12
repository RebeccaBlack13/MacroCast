"""US monthly targets (YoY CPI, core CPI, IP, M3) at 1/3/6/12-month horizons, walk-forward from 2000."""
import pickle, time, sys
from ml import *

TARGETS = ["cpi", "core", "ip", "m3"]
HORIZONS = [1, 3, 6, 12]
TEST_START = "2000-01-01"
START = {"cpi": "1962", "core": "1962", "ip": "1962", "m3": "1962"}

res, preds = [], {}
t0 = time.time()
for tg in TARGETS:
    for h in HORIZONS:
        for use_vix in (False, True):
            X, y, base = us_dataset(tg, h, vix=use_vix)
            own = [f"{tg}_{s}" for s in ("mom", "mom_l1", "g3", "g6", "yoy", "acc")]
            s = START[tg]
            P = walk_forward(X.loc[s:], y.loc[s:], model_zoo(own), test_start=TEST_START, h=h, refit_months=12)
            tag = f"{tg}|h{h}|{'vix' if use_vix else 'novix'}"
            preds[tag] = P
            # post-GFC / COVID-excluded view is computed in the report stage from stored predictions
            res += evaluate(P, h, tag)
            print(f"{tag:22s} done  {time.time()-t0:6.0f}s", flush=True)
pickle.dump(dict(res=pd.DataFrame(res), preds=preds), open("us_results.pkl", "wb"))
print("ALL DONE")
