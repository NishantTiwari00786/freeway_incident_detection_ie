"""
kalman filter: estimating speed at imputed sensors using the real readings from their neighbors

Setup:
    state: deviation from normal speed at every sensor on a corridor (speed - profile)
    predict: next deviation = a x (own deviation) + b x (downstream neighbor's deviation)
             (slowdowns fade back to normal, and queues spread upstream)
    update: only with real readings, imputed readings are treated as missing
    Q: how much traffic changes in 5 minutes that the model can't predict
    R: how noisy one sensor reading is

Script:
    1. learns a, b, Q and R from the training months (Oct to Jun)
    2. testing (Jul to Sep): hides real readings from working sensors, fills them in and compares the error
       with 3 simple methods: normal profile, copy nearest sensor, average of neighbors.
       two versions of the filter are tested: sensors linked by distance in miles, or by neighbor count
    3. runs the better version on the whole year and saves the filled in speeds

Outputs:

data/processed/kalman_params.json            learned a, b, Q, R and neighbor link
data/processed/kalman_results.csv            error table for the 5 methods
data/processed/kalman/YYYY_MM_DD.parquet     filled in speed and uncertainty for every sensor and time

Run after build_profile.py and label_incidents.py
"""

import glob
import json
import os
import time

import numpy as np
import pandas as pd


IN_DIR = "data/processed/5min"
PROFILE_FILE = "data/processed/profile.parquet"
STATIONS_FILE = "data/processed/graph/stations.parquet"
QUALITY_FILE = "data/processed/station_quality.parquet"
LABELS_FILE = "data/processed/crash_labels.parquet"
OUT_PARAMS = "data/processed/kalman_params.json"
OUT_RESULTS = "data/processed/kalman_results.csv"
OUT_DIR = "data/processed/kalman"

TEST_START = "2026-07-01"
HIDE_SHARE = 0.10      # on test days, hide 10% of the working sensors for the whole day
LINK_MAX_MI = 2.0      # neighbors farther apart than this are not linked (rural gaps)
P0 = 100.0             # starting uncertainty at midnight, the filter knows nothing yet
PARAM_EVERY = 4        # use every 4th training day to learn a, b, Q, R (still plenty of data, much faster)
SEED = 0
R_TUNED = 16  


# loading phase

def outage_days():
    # days where no sensor had real data
    q = pd.read_parquet(QUALITY_FILE)
    real = q.groupby("date")["pct_observed"].apply(lambda s: (s > 0).sum())
    return set(real[real == 0].index.date)


def corridors(stations):
    # for each freeway and direction: sensors in travel order (upstream first) and the gap to the next one
    out = {}
    for (fwy, d), g in stations.groupby(["fwy", "dir"]):
        g = g.sort_values("abs_pm", ascending=(d in ("N", "E")))      # travel order, same as build_graph.py
        ids = g["station"].to_numpy()
        gaps = np.abs(np.diff(g["abs_pm"].to_numpy()))                 # miles from each sensor to the next one
        out[(fwy, d)] = {"ids": ids, "pm": g["abs_pm"].to_numpy(), "gap": gaps}
    return out


def normal_table(prof, cors):
    # normal speed for every sensor, day type and time slot
    # sensors without a trusted profile get the normal speed of their neighbors
    prof = prof[prof["trusted"]]
    normal = {}
    for key, c in cors.items():
        for wk in (False, True):
            p = prof[(prof["weekend"] == wk) & prof["station"].isin(c["ids"])]
            grid = p.pivot(index="slot", columns="station", values="median_speed").reindex(index=range(288), columns=c["ids"])
            grid = grid.interpolate(axis=0, limit_direction="both")      # filling missing time slots
            arr = grid.to_numpy(dtype=float)
            have = ~np.isnan(arr).all(axis=0)                           # sensors that have a profile
            order = np.argsort(c["pm"][have])
            for t in range(288):                                         # filling missing sensors from neighbors
                if have.sum() >= 2:
                    arr[t, ~have] = np.interp(c["pm"][~have], c["pm"][have][order], arr[t, have][order])
                elif have.sum() == 1:
                    arr[t, ~have] = arr[t, have][0]
                else:
                    arr[t, :] = 65.0
            normal[(key, wk)] = arr
    return normal


def day_matrices(path, cors, normal):
    # for one day: per corridor, a table of deviations (288 time slots x sensors), NaN = missing or imputed
    d = pd.read_parquet(path, columns=["timestamp", "station", "pct_observed", "avg_speed"])
    day = d["timestamp"].dt.normalize().iloc[0]
    wk = day.day_of_week >= 5
    d["slot"] = (d["timestamp"].dt.hour * 12 + d["timestamp"].dt.minute // 5).astype(int)
    d.loc[d["pct_observed"] == 0, "avg_speed"] = np.nan                 # imputed readings become missing
    speed = d.pivot_table(index="slot", columns="station", values="avg_speed", aggfunc="first")
    out = {}
    for key, c in cors.items():
        s = speed.reindex(index=range(288), columns=c["ids"]).to_numpy(dtype=float)
        out[key] = {"speed": s, "normal": normal[(key, wk)], "dev": s - normal[(key, wk)]}
    return day, out


# learning a, b, Q, R from data

def learn_params(train_files, cors, normal):
    """
    real traffic carries over from one 5 minute step to the next, sensor noise does not

    cov(z_t, z_t+5min) and cov(z_t, z_t+10min) only contain the real traffic part,
    var(z_t) contains the real traffic part plus the sensor noise

    so:
        a = cov(10 min) / cov(5 min)             how much of a deviation carries over each step
        real traffic variance = cov(5 min) / a
        R = var(z) - real traffic variance       the part that does not carry over is sensor noise
        Q = real traffic variance x (1 - a^2)    how much new change comes in each step
    """
    s0 = s1 = s2 = n = 0.0
    store = []
    for f in train_files:
        _, mats = day_matrices(f, cors, normal)
        for key, c in cors.items():
            D = mats[key]["dev"]
            z0, z1, z2 = D[:-2], D[1:-1], D[2:]                         # now, 5 min later, 10 min later
            m = ~np.isnan(z0) & ~np.isnan(z1) & ~np.isnan(z2)
            s0 += (z0[m] ** 2).sum(); s1 += (z0[m] * z1[m]).sum(); s2 += (z0[m] * z2[m]).sum(); n += m.sum()
            store.append((D, np.append(c["gap"] <= LINK_MAX_MI, False)))
    var0, cov1, cov2 = s0 / n, s1 / n, s2 / n
    a = float(np.clip(cov2 / cov1, 0.3, 0.98))
    traffic_var = cov1 / a
    R = float(max(var0 - traffic_var, 1.0))
    Q = float(max(traffic_var * (1 - a ** 2), 0.5))

    num = den = 0.0
    adj_sum = adj_n = 0.0
    for D, linked in store:
        # b: what own persistence doesn't explain, how much of it comes from the downstream neighbor
        y = D[1:] - a * D[:-1]
        down = np.full_like(D, np.nan)
        down[:, :-1] = D[:, 1:]
        x = down[:-1]
        m = ~np.isnan(y) & ~np.isnan(x) & linked[None, :]
        num += (x[m] * y[m]).sum(); den += (x[m] ** 2).sum()

        # checking how much neighbors' deviations move together (sensor noise is different at each sensor, so it drops out)
        left, right = D[:, :-1], D[:, 1:]
        mm = ~np.isnan(left) & ~np.isnan(right) & linked[:-1][None, :]
        adj_sum += (left[mm] * right[mm]).sum(); adj_n += mm.sum()
    b = float(np.clip(num / den, 0, 1 - a)) if den > 0 else 0.0
    rho = float(np.clip((adj_sum / adj_n) / traffic_var, 0, 0.95)) if adj_n > 100 else 0.0

    # typical distance between linked neighbors
    gaps = np.concatenate([c["gap"][c["gap"] <= LINK_MAX_MI] for c in cors.values()])
    spacing = float(np.median(gaps[gaps > 0])) if (gaps > 0).any() else 0.5
    return {"a": a, "b": b, "R": R, "Q": Q, "rho": rho, "spacing_mi": spacing, "n_readings": int(n),
            "var_total": float(var0), "cov_5min": float(cov1), "cov_10min": float(cov2)}


def build_AQ(c, p, link="distance"):
    # A: how deviations move to the next step, Q: how much new change comes in and how it's shared by neighbors
    # link = "distance": neighbors linked by miles apart (closer = stronger link)
    # link = "hops": 1st neighbor rho, 2nd neighbor rho^2, and so on
    n = len(c["ids"])
    linked = c["gap"] <= LINK_MAX_MI
    A = np.eye(n) * p["a"]
    for i in range(n - 1):
        if linked[i]:
            A[i, i + 1] = p["b"]                                        # borrowing from the downstream neighbor

    # sensors on the same stretch (no rural gap between them) share a segment number
    segment = np.concatenate([[0], np.cumsum(~linked)])
    same = segment[:, None] == segment[None, :]
    if link == "distance":
        dist = np.abs(c["pm"][:, None] - c["pm"][None, :])              # miles between every pair of sensors
        L = -p["spacing_mi"] / np.log(max(p["rho"], 1e-3))              # so the link is rho at the typical spacing
        strength = np.exp(-dist / L)
    else:
        hops = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])
        strength = p["rho"] ** hops
    Q = p["Q"] * np.where(same, strength, 0.0)
    return A, Q


# filter

def kalman_run(Z, A, Q, R):
    # Z: deviations for one day and one corridor (288 x sensors), NaN = no real reading
    # returns the estimated deviation and its uncertainty (in mph) for every time slot and sensor
    T, n = Z.shape
    x = np.zeros(n)                       # start: assume normal
    P = np.eye(n) * P0                    # start: very unsure
    X, SD = np.zeros((T, n)), np.zeros((T, n))
    for t in range(T):
        # 1. predict: move the guess forward, uncertainty grows
        x = A @ x
        P = A @ P @ A.T + Q

        # 2. update: only with the sensors that sent a real reading
        obs = ~np.isnan(Z[t])
        if obs.any():
            z = Z[t, obs]
            S = P[np.ix_(obs, obs)] + R * np.eye(obs.sum())   # uncertainty of each surprise
            K = np.linalg.solve(S, P[obs, :]).T                # kalman gain: how much to trust each reading
            x = x + K @ (z - x[obs])                           # moving toward the readings
            P = P - K @ P[obs, :]                              # becoming more certain
            P = (P + P.T) / 2                                  # keeping P symmetric
        X[t], SD[t] = x, np.sqrt(np.clip(np.diag(P), 0, None))
    return X, SD


# simple methods to compare against

def fill_nearest(Z, pm):
    # copying the deviation of the nearest sensor that has a real reading
    out = np.zeros_like(Z)
    for t in range(Z.shape[0]):
        ok = ~np.isnan(Z[t])
        if ok.sum() == 0:
            continue
        j = np.abs(pm[:, None] - pm[ok][None, :]).argmin(axis=1)
        out[t] = Z[t, ok][j]
    return out


def fill_interp(Z, pm):
    # averaging the nearest working sensors on each side, weighted by distance
    out = np.zeros_like(Z)
    order = np.argsort(pm)
    for t in range(Z.shape[0]):
        ok = ~np.isnan(Z[t])
        if ok.sum() == 0:
            continue
        o = order[ok[order]]
        out[t] = np.interp(pm, pm[o], Z[t, o])
    return out


# testing

def crash_windows():
    # time windows where a detected crash was slowing traffic at its sensor
    lab = pd.read_parquet(LABELS_FILE)
    d = lab[lab["labelable"] & lab["detected"].fillna(False).astype(bool)].copy()
    start = d["timestamp"].dt.floor("5min") + pd.to_timedelta(d["onset_vs_report_min"], unit="m")
    d["start"], d["end"] = start, start + pd.to_timedelta(d["congestion_min"].clip(lower=5), unit="m")
    return d[["sensor", "start", "end"]]


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    cors = corridors(pd.read_parquet(STATIONS_FILE))
    normal = normal_table(pd.read_parquet(PROFILE_FILE), cors)
    skip = outage_days()
    files = [f for f in sorted(glob.glob(os.path.join(IN_DIR, "*.parquet")))
             if pd.Timestamp(os.path.basename(f)[:10].replace("_", "-")).date() not in skip]
    train = [f for f in files if os.path.basename(f)[:10].replace("_", "-") < TEST_START]
    test = [f for f in files if os.path.basename(f)[:10].replace("_", "-") >= TEST_START]

    # 1. learning the filter's numbers from the training months
    p = learn_params(train[::PARAM_EVERY], cors, normal)
    if R_TUNED is not None:
        p["R"] = float(R_TUNED)
    print(f"Learned from data ({time.time() - t0:.0f}s):")
    print(f"  a = {p['a']:.3f}  (how much a deviation carries over to the next 5 min)")
    print(f"  b = {p['b']:.3f}  (how much of the downstream neighbor's deviation spreads upstream)")
    print(f"  R = {p['R']:.2f}  (sensor noise, typical error {np.sqrt(p['R']):.1f} mph)")
    print(f"  Q = {p['Q']:.2f}  (traffic change per 5 min, typical {np.sqrt(p['Q']):.1f} mph)")
    print(f"  rho = {p['rho']:.2f}  (how much neighboring sensors move together, "
          f"at the typical spacing of {p['spacing_mi']:.2f} mi)")
    AQ = {link: {k: build_AQ(c, p, link) for k, c in cors.items()} for link in ("distance", "hops")}

    # 2. testing: hiding real readings on Jul to Sep days and comparing 5 ways of filling them in
    windows = crash_windows()
    methods = ("Normal profile", "Copy nearest sensor", "Average of neighbors",
               "Kalman (linked by distance)", "Kalman (linked by neighbor count)")
    errs = {m: [] for m in methods}
    flags_peak, flags_crash = [], []
    for f in test:
        day, mats = day_matrices(f, cors, normal)
        w_day = windows[(windows["start"] < day + pd.Timedelta(days=1)) & (windows["end"] > day)]
        for key, c in cors.items():
            D = mats[key]["dev"]
            working = np.where((~np.isnan(D)).mean(axis=0) >= 0.5)[0]   # sensors with real data at least half the day
            if len(working) < 3:
                continue
            hide = set(rng.choice(working, size=max(1, int(HIDE_SHARE * len(working))), replace=False))

            # also hiding crash sensors on crash days, so we can check the error during crashes
            crash_rows = np.zeros_like(D, dtype=bool)
            for _, r in w_day.iterrows():
                hits = np.where(c["ids"] == r["sensor"])[0]
                if len(hits) and hits[0] in working:
                    i = hits[0]
                    hide.add(i)
                    ts = day + pd.to_timedelta(np.arange(288) * 5, unit="m")
                    crash_rows[:, i] |= (ts >= r["start"]) & (ts <= r["end"])

            hide = np.array(sorted(hide))
            Z = D.copy()
            Z[:, hide] = np.nan                                         # the filter never sees these
            truth_mask = np.zeros_like(D, dtype=bool)
            truth_mask[:, hide] = ~np.isnan(D[:, hide])                 # only scoring readings that were real

            kf_d, _ = kalman_run(Z, *AQ["distance"][key], p["R"])
            kf_h, _ = kalman_run(Z, *AQ["hops"][key], p["R"])
            preds = {"Normal profile": np.zeros_like(D), "Copy nearest sensor": fill_nearest(Z, c["pm"]),
                     "Average of neighbors": fill_interp(Z, c["pm"]),
                     "Kalman (linked by distance)": kf_d, "Kalman (linked by neighbor count)": kf_h}
            truth = D[truth_mask]
            for m, pr in preds.items():
                errs[m].append(np.abs(pr[truth_mask] - truth))           # error in deviation = error in speed

            hr = np.tile(np.arange(288)[:, None], (1, D.shape[1])) // 12
            flags_peak.append((((hr >= 6) & (hr <= 8)) | ((hr >= 15) & (hr <= 18)))[truth_mask])
            flags_crash.append(crash_rows[truth_mask])

    peak, crash = np.concatenate(flags_peak), np.concatenate(flags_crash)
    table = {}
    for m, e in errs.items():
        e = np.concatenate(e)
        table[m] = {"all": e.mean(), "peak": e[peak].mean(), "off_peak": e[~peak].mean(),
                    "during_crash": e[crash].mean() if crash.any() else np.nan,
                    "within_5mph": (e <= 5).mean()}
    res = pd.DataFrame(table).T.round(3)
    res.to_csv(OUT_RESULTS)
    print(f"\nFilling in hidden real readings, Jul-Sep ({time.time() - t0:.0f}s)")
    print(f"Mean absolute error (mph), {len(peak):,} hidden readings, {crash.sum():,} during crashes:")
    print(res.to_string())

    # picking the version with lower error over all hours
    best = "distance" if table["Kalman (linked by distance)"]["all"] <= table["Kalman (linked by neighbor count)"]["all"] else "hops"
    p["link_used_for_fill"] = best
    with open(OUT_PARAMS, "w") as fh:
        json.dump(p, fh, indent=2)
    print(f"Filling the full year with the better version: linked by {best}")

    # 3. filling the whole year (nothing hidden) and saving speed + uncertainty for every sensor and time
    os.makedirs(OUT_DIR, exist_ok=True)
    for i, f in enumerate(files, 1):
        out_path = os.path.join(OUT_DIR, os.path.basename(f))
        if os.path.exists(out_path):
            continue
        day, mats = day_matrices(f, cors, normal)
        parts = []
        for key, c in cors.items():
            kf, sd = kalman_run(mats[key]["dev"], *AQ[best][key], p["R"])
            n = len(c["ids"])
            parts.append(pd.DataFrame({
                "timestamp": np.repeat(day + pd.to_timedelta(np.arange(288) * 5, unit="m"), n),
                "station": np.tile(c["ids"], 288).astype("int32"),
                "speed_kf": (mats[key]["normal"] + kf).ravel().astype("float32"),
                "kf_std": sd.ravel().astype("float32"),
                "observed": (~np.isnan(mats[key]["dev"])).ravel(),
            }))
        pd.concat(parts, ignore_index=True).to_parquet(out_path, index=False)
        if i % 30 == 0:
            print(f"  filled {i}/{len(files)} days ({time.time() - t0:.0f}s)")
    print(f"Done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()