"""
tuning R for the kalman filter

R landed at its lower limit (1.0), so the filter trusts every reading almost completely.
this script tries a few R values and checks if a larger R works better.

Script:
    1. learns a, b, Q, rho the same way as kalman_fill.py
    2. tries each R on June (the last training month), using the same hide-and-fill test
    3. picks the R with the lowest error on June
    4. checks that R once on Jul to Sep, next to R = 1.0

we pick R on June, not on Jul to Sep, so the final test stays fair (we never tune on the test data)

Run after kalman_fill.py
"""

import glob
import os

import numpy as np
import pandas as pd

from kalman_fill import (IN_DIR, PROFILE_FILE, STATIONS_FILE, TEST_START, HIDE_SHARE, PARAM_EVERY, SEED,
                         outage_days, corridors, normal_table, day_matrices, learn_params, build_AQ,
                         kalman_run, crash_windows)

R_VALUES = [1, 2, 4, 8, 16, 32]
VALID_START = "2026-06-01"
LINK = "hops"          # the version kalman_fill.py picked


def date_of(f):
    return os.path.basename(f)[:10].replace("_", "-")


def evaluate(files, cors, normal, AQ, R_list, windows):
    # same test as kalman_fill.py: hide 10% of working sensors (plus crash sensors) for the whole day
    # the same readings are hidden for every R, so the comparison is fair
    rng = np.random.default_rng(SEED)
    errs = {R: [] for R in R_list}
    crash_flags = []
    for f in files:
        day, mats = day_matrices(f, cors, normal)
        w_day = windows[(windows["start"] < day + pd.Timedelta(days=1)) & (windows["end"] > day)]
        for key, c in cors.items():
            D = mats[key]["dev"]
            working = np.where((~np.isnan(D)).mean(axis=0) >= 0.5)[0]
            if len(working) < 3:
                continue
            hide = set(rng.choice(working, size=max(1, int(HIDE_SHARE * len(working))), replace=False))
            crash_rows = np.zeros_like(D, dtype=bool)
            for _, r in w_day.iterrows():
                hits = np.where(c["ids"] == r["sensor"])[0]
                if len(hits) and hits[0] in working:
                    hide.add(hits[0])
                    ts = day + pd.to_timedelta(np.arange(288) * 5, unit="m")
                    crash_rows[:, hits[0]] |= (ts >= r["start"]) & (ts <= r["end"])
            hide = np.array(sorted(hide))
            Z = D.copy()
            Z[:, hide] = np.nan
            truth_mask = np.zeros_like(D, dtype=bool)
            truth_mask[:, hide] = ~np.isnan(D[:, hide])
            for R in R_list:
                kf, _ = kalman_run(Z, *AQ[key], R)
                errs[R].append(np.abs(kf[truth_mask] - D[truth_mask]))
            crash_flags.append(crash_rows[truth_mask])
    crash = np.concatenate(crash_flags)
    rows = {}
    for R in R_list:
        e = np.concatenate(errs[R])
        rows[R] = {"all": e.mean(), "during_crash": e[crash].mean() if crash.any() else np.nan}
    return pd.DataFrame(rows).T.round(3), int(crash.sum())


def main():
    cors = corridors(pd.read_parquet(STATIONS_FILE))
    normal = normal_table(pd.read_parquet(PROFILE_FILE), cors)
    skip = outage_days()
    files = [f for f in sorted(glob.glob(os.path.join(IN_DIR, "*.parquet")))
             if pd.Timestamp(date_of(f)).date() not in skip]
    train = [f for f in files if date_of(f) < TEST_START]
    valid = [f for f in files if VALID_START <= date_of(f) < TEST_START]
    test = [f for f in files if date_of(f) >= TEST_START]

    p = learn_params(train[::PARAM_EVERY], cors, normal)
    AQ = {k: build_AQ(c, p, LINK) for k, c in cors.items()}
    windows = crash_windows()

    # 1. trying each R on June
    res, n_crash = evaluate(valid, cors, normal, AQ, R_VALUES, windows)
    print(f"June (validation), mean absolute error in mph, {n_crash:,} readings during crashes:")
    print(res.to_string())
    best = res["all"].idxmin()
    print(f"\nBest R on June: {best}")

    # 2. checking the chosen R once on Jul to Sep, next to the current R = 1.0
    check = sorted({1, best})
    res_t, n_crash_t = evaluate(test, cors, normal, AQ, check, windows)
    print(f"\nJul-Sep (test), mean absolute error in mph, {n_crash_t:,} readings during crashes:")
    print(res_t.to_string())


if __name__ == "__main__":
    main()