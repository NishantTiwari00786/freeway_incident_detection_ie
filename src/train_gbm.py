"""
Gradient Boosting model to forecast speed 15 minutes ahead at each sensor 
    
I used the build_profile.py which only knows what is Normal traffic for a sensor at time of a day. 
It can't change the numbers by prediciting what is happening right now (crash)
The model uses the current traffic + normal profile to forecast the speed 15 minutes from now 
    
I compared 4 forecasts on same held-out data (Jul to Sep):
persistence = speed in 15 min = speed now (the simplest possible forecast)
profile = speed in 15 min = normal speed for that sensor and time 
GBM (own sensor): gradient boosting using only the sensor's own recent speeds + profile + time features
GBM (+ neighbors): the same + plus the current speeds at upstream and downstream sensors
    
Each forecast is scored for all readings, peak hours, and during crashes 
    
Script: 
1. builds features for every day: recent speeds (lags), change in speed, normal speed, time of day,
and neighbor speeds from the sensor graph
2. keeps only real readings (no imputed values as inputs or targets), and samples rows so it fits in memory
3. trains on Oct to Jun, tests on Jul to Sep
4. saves the scores and the trained model
    
    
    Outputs: 
    data/processed/gbm_results.csv error table for all forecasts
    data/processed/gbm_model.joblib  the trained model (with neighbors)
    
    It works after running build_profile.py and label_incidents.py
    

"""


import glob
import os
import time
 
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
 
 
IN_DIR = "data/processed/5min"
PROFILE_FILE = "data/processed/profile.parquet"
EDGES_FILE = "data/processed/graph/edges.parquet"
QUALITY_FILE = "data/processed/station_quality.parquet"
LABELS_FILE = "data/processed/crash_labels.parquet"
OUT_RESULTS = "data/processed/gbm_results.csv"
OUT_MODEL = "data/processed/gbm_model.joblib"
OUT_IMPORTANCE = "data/processed/gbm_feature_importance.csv"
OUT_FIG = "figures/fig6_feature_importance.png"
 
HORIZON = 3              # 3 slots of 5 min = forecast 15 minutes ahead
LAGS = [0, 1, 2, 3, 6]   # speed now, 5, 10, 15, and 30 minutes ago
TEST_START = "2026-07-01"
SAMPLE = 0.15            # keep 15% of normal rows per day so the full year fits in memory (crash rows are always kept)
SEED = 0
 
OWN = ["speed_lag0", "speed_lag1", "speed_lag2", "speed_lag3", "speed_lag6", "change_15", "occ_lag0",
       "flow_lag0", "normal_now", "normal_target", "slot", "weekend"]
# neighbors: up1 = next sensor upstream, up2 = the one after that, same for downstream
HOPS = ["up1", "up2", "down1", "down2"]
NB1 = ["up1_speed", "up1_gap", "down1_speed", "down1_gap"]              # 1 neighbor each way, now only
NB2 = [f"{h}_{f}" for h in HOPS for f in ("speed", "speed_5ago", "speed_15ago", "gap")]   # 2 each way + history
 
 
def outage_days():
    q = pd.read_parquet(QUALITY_FILE)
    real = q.groupby("date")["pct_observed"].apply(lambda s: (s > 0).sum())
    return set(real[real == 0].index.date)
 
 
def crash_windows():
    """Time windows where a detected crash was congesting its sensor, used to score forecasts during crashes."""
    lab = pd.read_parquet(LABELS_FILE)
    d = lab[lab["labelable"] & lab["detected"].fillna(False).astype(bool)].copy()
    start = d["timestamp"].dt.floor("5min") + pd.to_timedelta(d["onset_vs_report_min"], unit="m")
    d["start"], d["end"] = start, start + pd.to_timedelta(d["congestion_min"].clip(lower=5), unit="m")
    return d[["sensor", "start", "end"]].rename(columns={"sensor": "station"})
 
 
def build_day(path, prof, hops, windows):
    d = pd.read_parquet(path, columns=["timestamp", "station", "pct_observed", "avg_speed", "avg_occupancy", "total_flow"])
    d = d.sort_values(["station", "timestamp"])
    d.loc[d["pct_observed"] == 0, ["avg_speed", "avg_occupancy", "total_flow"]] = np.nan   # imputed values become missing
    d["weekend"] = d["timestamp"].dt.day_of_week >= 5
    d["slot"] = (d["timestamp"].dt.hour * 12 + d["timestamp"].dt.minute // 5).astype("int16")
 
    g = d.groupby("station")["avg_speed"]
    for k in LAGS:
        d[f"speed_lag{k}"] = g.shift(k)
    d["change_15"] = d["speed_lag0"] - d["speed_lag3"]
    d["occ_lag0"] = d["avg_occupancy"]
    d["flow_lag0"] = d["total_flow"]
    d["target"] = g.shift(-HORIZON)                       # what we are trying to predict: speed 15 min from now
    d["target_slot"] = (d["slot"] + HORIZON).clip(upper=287)
 
    # normal speed now and at the target time (from the profile)
    p = prof.rename(columns={"median_speed": "normal_now"})
    d = d.merge(p, on=["station", "weekend", "slot"], how="left")
    p = prof.rename(columns={"slot": "target_slot", "median_speed": "normal_target"})
    d = d.merge(p, on=["station", "weekend", "target_slot"], how="left")
 
    # neighbor features: for each of the 4 neighbors, its speed now, 5 and 15 min ago,
    # and its gap from normal right now (negative = slower than usual)
    now = d[["station", "timestamp", "speed_lag0", "speed_lag1", "speed_lag3"]].copy()
    now["gap"] = d["speed_lag0"] - d["normal_now"]
    for hop, links in hops.items():
        nb = links.merge(now.rename(columns={"station": "nb", "speed_lag0": f"{hop}_speed", "speed_lag1": f"{hop}_speed_5ago",
                                             "speed_lag3": f"{hop}_speed_15ago", "gap": f"{hop}_gap"}), on="nb", how="inner")
        d = d.merge(nb.drop(columns="nb"), on=["station", "timestamp"], how="left")
 
    d = d[d["target"].notna() & d["speed_lag0"].notna() & d["normal_target"].notna()]
 
    # mark rows inside a crash window
    d["in_crash"] = False
    w = windows[(windows["start"] <= d["timestamp"].max()) & (windows["end"] >= d["timestamp"].min())]
    for _, r in w.iterrows():
        m = (d["station"] == r["station"]) & d["timestamp"].between(r["start"], r["end"])
        d.loc[m, "in_crash"] = True
 
    keep = d["in_crash"] | (np.random.default_rng(SEED + len(d)).random(len(d)) < SAMPLE)
    out = d.loc[keep, OWN + NB2 + ["target", "timestamp", "station", "in_crash"]]
    num = OWN + NB2 + ["target"]
    out[num] = out[num].astype("float32")      # float32 halves memory for ~7 million rows x 30 columns
    return out
 
 
def scores(y, pred, mask_peak, mask_crash):
    e = np.abs(y - pred)
    return {"all": e.mean(), "peak": e[mask_peak].mean(), "off_peak": e[~mask_peak].mean(), "during_crash": e[mask_crash].mean()}
 
 
def plot_importance(imp, top=12):
    """Bar chart of the most important features: how many mph the error grows when each one is shuffled."""
    nice = {"speed_lag0": "Speed now", "speed_lag1": "Speed 5 min ago", "speed_lag2": "Speed 10 min ago",
            "speed_lag3": "Speed 15 min ago", "speed_lag6": "Speed 30 min ago", "change_15": "Change in last 15 min",
            "occ_lag0": "Occupancy now", "flow_lag0": "Flow now", "normal_now": "Normal speed now",
            "normal_target": "Normal speed in 15 min", "slot": "Time of day", "weekend": "Weekend"}
    for h, word in (("up1", "1st upstream"), ("up2", "2nd upstream"), ("down1", "1st downstream"), ("down2", "2nd downstream")):
        nice.update({f"{h}_speed": f"{word}: speed now", f"{h}_speed_5ago": f"{word}: speed 5 min ago",
                     f"{h}_speed_15ago": f"{word}: speed 15 min ago", f"{h}_gap": f"{word}: gap from normal"})
    os.makedirs(os.path.dirname(OUT_FIG), exist_ok=True)
    cols = list(imp.columns)
    fig, axes = plt.subplots(1, len(cols), figsize=(5.2 * len(cols), 4.6), dpi=160)
    axes = np.atleast_1d(axes)
    for ax, c in zip(axes, cols):
        s = imp[c].sort_values(ascending=False).head(top)[::-1]
        colors = ["#256abf" if not k.startswith(("up", "down")) else "#86b6ef" for k in s.index]
        ax.barh([nice.get(k, k) for k in s.index], s.values, color=colors, height=0.65)
        ax.set_title(f"Top features, {c}", loc="left", fontsize=10, color="#1f1f1e")
        ax.set_xlabel("Error increase when shuffled (mph)", color="#6b6a66", fontsize=8.5)
        ax.tick_params(labelsize=8, colors="#6b6a66")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.xaxis.grid(True, color="#ecebe7"); ax.set_axisbelow(True)
    fig.text(0.01, -0.02, "Dark blue = the sensor's own features, light blue = neighbor features. "
             "Permutation importance on held-out Jul-Sep data.", fontsize=7.5, color="#6b6a66")
    fig.tight_layout()
    fig.savefig(OUT_FIG, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {OUT_FIG}")
 
 
def main():
    t0 = time.time()
    prof = pd.read_parquet(PROFILE_FILE)
    prof = prof[prof["trusted"]][["station", "weekend", "slot", "median_speed"]]
    edges = pd.read_parquet(EDGES_FILE)
    up1 = edges.rename(columns={"downstream": "station", "upstream": "nb"})[["station", "nb"]]     # my upstream neighbor
    down1 = edges.rename(columns={"upstream": "station", "downstream": "nb"})[["station", "nb"]]   # my downstream neighbor
    # 2 hops: my upstream neighbor's upstream neighbor (same idea downstream)
    up2 = up1.merge(up1.rename(columns={"station": "nb", "nb": "nb2"}), on="nb")[["station", "nb2"]].rename(columns={"nb2": "nb"})
    down2 = down1.merge(down1.rename(columns={"station": "nb", "nb": "nb2"}), on="nb")[["station", "nb2"]].rename(columns={"nb2": "nb"})
    hops = {"up1": up1, "up2": up2, "down1": down1, "down2": down2}
    windows = crash_windows()
    skip = outage_days()
 
    parts = []
    files = sorted(glob.glob(os.path.join(IN_DIR, "*.parquet")))
    for i, f in enumerate(files, 1):
        day = pd.Timestamp(os.path.basename(f)[:10].replace("_", "-"))
        if day.date() in skip:
            continue
        parts.append(build_day(f, prof, hops, windows))
        if i % 30 == 0:
            print(f"  features: {i}/{len(files)} days ({time.time() - t0:.0f}s)")
    data = pd.concat(parts, ignore_index=True)
    data["weekend"] = data["weekend"].astype(int)
    print(f"{len(data):,} rows, {data['in_crash'].sum():,} inside crash windows")
 
    train = data[data["timestamp"] < TEST_START]
    test = data[data["timestamp"] >= TEST_START]
    print(f"train {len(train):,} rows (Oct-Jun) | test {len(test):,} rows (Jul-Sep)")
 
    y = test["target"].to_numpy()
    hour = test["timestamp"].dt.hour
    peak = (hour.between(6, 8) | hour.between(15, 18)).to_numpy()
    crash = test["in_crash"].to_numpy()
 
    results = {
        "persistence": scores(y, test["speed_lag0"].to_numpy(), peak, crash),
        "profile": scores(y, test["normal_target"].to_numpy(), peak, crash),
    }
    models = (("GBM (own sensor)", OWN), ("GBM (+1 neighbor, now)", OWN + NB1),
              ("GBM (+2 neighbors, history)", OWN + NB2))
    for name, feats in models:
        m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.1, max_leaf_nodes=63,
                                          early_stopping=True, validation_fraction=0.1, random_state=SEED)
        m.fit(train[feats], train["target"])
        results[name] = scores(y, m.predict(test[feats]), peak, crash)
        print(f"  trained {name} ({time.time() - t0:.0f}s)")
    joblib.dump(m, OUT_MODEL)      # the last one: 2 neighbors with history
 
    # feature importance on the final model: shuffle one feature at a time and see how much the error grows.
    # done on a sample of normal test rows, and separately on crash rows, because the model may rely on
    # different features when traffic is normal vs during a crash
    feats = OWN + NB2
    rng = np.random.default_rng(SEED)
    normal_rows = test[~test["in_crash"]].sample(n=min(50_000, (~test["in_crash"]).sum()), random_state=SEED)
    crash_rows = test[test["in_crash"]]
    imp = {}
    for label, rows in (("all hours", normal_rows), ("during crashes", crash_rows)):
        if len(rows) < 50:
            continue
        r = permutation_importance(m, rows[feats], rows["target"], scoring="neg_mean_absolute_error",
                                   n_repeats=3, random_state=SEED)
        imp[label] = pd.Series(r.importances_mean, index=feats)
    imp = pd.DataFrame(imp)
    imp.round(3).to_csv(OUT_IMPORTANCE)
    print(f"  feature importance done ({time.time() - t0:.0f}s)")
    plot_importance(imp)
 
    res = pd.DataFrame(results).T.round(2)
    res.to_csv(OUT_RESULTS)
    print(f"\nMean absolute error (mph), forecasting {HORIZON * 5} min ahead, test Jul-Sep:")
    print(res.to_string())
    print(f"\nTest rows: {len(test):,} | peak: {peak.sum():,} | during crashes: {crash.sum():,}")
    print("\nTop 8 features during crashes (mph of error added when shuffled):")
    if "during crashes" in imp:
        print(imp["during crashes"].sort_values(ascending=False).head(8).round(2).to_string())
    print(f"Done in {time.time() - t0:.0f}s")
 
 
if __name__ == "__main__":
    main()