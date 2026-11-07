"""
Building the expected traffic profile: what normal speed look like at each sensor 

for each sensor, day type (week or weekend) and using 5-minute time of day, I took the Median 
of speed over the whole year, using only real (observed) readings. The median is helpful then mean because mean 
can drag the numbers down due to few crashes

This script: 
    1. reads all daily files, keeping only observed rows and skipping the synthetic data added by pems 
    2. computes median speed per (station, day type, time slot) and how many days it is based on 
    3. scores the profile as baseline: how far off is normal from the actual speed on held out days 
    
Output:
data/processed/profile.parquet   one row per (station, weekend, slot) with median_speed and n_days
 
This file would only work if we run it after process_5min.oy


"""


import glob
import os
 
import numpy as np
import pandas as pd
 
 
IN_DIR = "data/processed/5min"
QUALITY_FILE = "data/processed/station_quality.parquet"
OUT_FILE = "data/processed/profile.parquet"
 
MIN_DAYS = 10          # a profile value based on fewer real days t
TEST_START = "2026-07-01"   # Jul to Sep is held out to score the profile as a baseline
 
 
def outage_days():
    # days where no sensor had real data, same rule as load_incidents.py
    q = pd.read_parquet(QUALITY_FILE)
    real = q.groupby("date")["pct_observed"].apply(lambda s: (s > 0).sum())
    return set(real[real == 0].index.date)
 
 
def add_keys(df):
    # weekend flag and 5 minute slot of the day (0 = 00:00, 287 = 23:55)
    # stored as small numbers instead of text, to keep ~55 million rows in memory
    df["weekend"] = df["timestamp"].dt.dayofweek >= 5
    df["slot"] = (df["timestamp"].dt.hour * 12 + df["timestamp"].dt.minute // 5).astype("int16")
    return df.drop(columns=["timestamp", "pct_observed"])
 
 
def load(files, skip):
    parts = []
    for f in files:
        day = pd.Timestamp(os.path.basename(f)[:10].replace("_", "-")).date()
        if day in skip:
            continue
        d = pd.read_parquet(f, columns=["timestamp", "station", "pct_observed", "avg_speed"])
        d = d[d["pct_observed"] > 0]           # real readings only
        parts.append(add_keys(d))
    return pd.concat(parts, ignore_index=True)
 
 
def main():
    files = sorted(glob.glob(os.path.join(IN_DIR, "*.parquet")))
    skip = outage_days()
    print(f"{len(files)} days found, skipping {len(skip)} outage days")
 
    train_files = [f for f in files if os.path.basename(f)[:10].replace("_", "-") < TEST_START]
    test_files = [f for f in files if os.path.basename(f)[:10].replace("_", "-") >= TEST_START]
 
   # score the profile as a baseline: build it on Oct to Jun, test on Jul to Sep
    train = load(train_files, skip)
    prof_train = (train.groupby(["station", "weekend", "slot"])["avg_speed"]
                       .agg(median_speed="median", n_days="count").reset_index())
    prof_train = prof_train[prof_train["n_days"] >= MIN_DAYS]
    del train
 
    test = load(test_files, skip)
    test = test.merge(prof_train, on=["station", "weekend", "slot"], how="inner")
    err = (test["avg_speed"] - test["median_speed"]).abs()
    peak = test["slot"].between(6 * 12, 9 * 12 - 1) | test["slot"].between(15 * 12, 19 * 12 - 1)
    print(f"\nHistorical-median baseline (built Oct-Jun, tested Jul-Sep, {len(test):,} readings):")
    print(f"  MAE all hours: {err.mean():.2f} mph | peak hours: {err[peak].mean():.2f} mph | off-peak: {err[~peak].mean():.2f} mph")
    print(f"  Readings off by more than 20 mph: {(err > 20).mean():.1%}")
    del test
 
    # final profile on the full year, used for incident labels
    full = load(files, skip)
    prof = (full.groupby(["station", "weekend", "slot"])["avg_speed"]
                .agg(median_speed="median", n_days="count").reset_index())
    prof["median_speed"] = prof["median_speed"].astype("float32")
    prof["trusted"] = prof["n_days"] >= MIN_DAYS
    prof.to_parquet(OUT_FILE, index=False)
 
    all_stations = pd.read_parquet(QUALITY_FILE)["station"].nunique()
    st = prof[prof["trusted"]].groupby("station").size()
    print(f"\nProfile: {len(prof):,} rows")
    print(f"Stations with a trusted profile for all 576 slots (288 weekday + 288 weekend): {(st == 576).sum()} of {all_stations}")
    print(f"Stations with some trusted slots: {len(st)} | with none (no usable real data): {all_stations - len(st)}")
 
 
if __name__ == "__main__":
    main()
 