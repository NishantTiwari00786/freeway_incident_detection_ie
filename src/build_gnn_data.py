"""

GNN daa: puts the whole year into a few big arrays (time x sensors) so the GNN can trainf ast

Since GNN doesnt read 365 files while training, 
    speed_kf = speed notes that come from Kalman filter (input)
    kf_std = how unsure kalman is (small val = very close approximity reading)
    real = the real speed, NaN where PeMS was guessing (what I score on)
    normal = normal speed from profile of that sensor, day type and time slot 
    in_crash = true while a detected crash was congesting that sensor 
    
Sensors are in same orders as kf corridors 

Outputs: 
data/processed/gnn/gnn_data.npz # the array + times, stations, slot, weekend 

Run after kalman_fill.py

"""
import glob
import os 

import numpy as np 
import pandas as pd 

from kalman_fill import corridors, crash_windows, day_matrices, normal_table, outage_days

KALMAN_DIR = "data/processed/kalman"
IN_DIR = "data/processed/5min"
PROFILE_FILE = "data/processed/profile.parquet"
STATIONS_FILE = "data/processed/graph/stations.parquet"
OUT_DIR = "data/processed/gnn"


def main():
    cors = corridors(pd.read_parquet(STATIONS_FILE))
    normal = normal_table(pd.read_parquet(PROFILE_FILE), cors)
    ids = np.concatenate([c["ids"] for c in cors.values()]) # one fixed sensor order
    skip = outage_days()

    files = sorted(glob.glob(os.path.join(KALMAN_DIR, "*.parquet")))
    print(f"{len(files)} kalman days, {len(ids)} sensors")

    kf, std, real, norm, times = [], [], [], [], []
    for i, f in enumerate(files, 1):
        name = os.path.basename(f)
        day = pd.Timestamp(name[:10].replace("_", "-"))
        if day.date() in skip or not os.path.exists(os.path.join(IN_DIR, name)):
            continue

        # real speed and normal speed, from the same helpers kalman uses
        _, mats = day_matrices(os.path.join(IN_DIR, name), cors, normal)
        r = np.hstack([mats[k]["speed"] for k in cors])               # 288 x sensors, NaN = imputed
        n = np.hstack([mats[k]["normal"] for k in cors])

        # kalman speed as a 288 x sensors table
        k = pd.read_parquet(f)
        k["slot"] = k["timestamp"].dt.hour * 12 + k["timestamp"].dt.minute // 5
        s = k.pivot_table(index="slot", columns="station", values="speed_kf").reindex(index=range(288), columns=ids)
        u = k.pivot_table(index="slot", columns="station", values="kf_std").reindex(index=range(288), columns=ids)
        s = s.to_numpy(dtype=float)
        s = np.where(np.isnan(s), n, s)                               # sensor missing from kalman -> normal speed

        kf.append(s.astype("float16"))
        std.append(u.fillna(u.max().max()).to_numpy().astype("float16"))
        real.append(r.astype("float16"))
        norm.append(n.astype("float16"))
        times.append(day + pd.to_timedelta(np.arange(288) * 5, unit="min"))
        if i % 30 == 0:
            print(f"  {i}/{len(files)} days")

    kf, std, real, norm = (np.concatenate(a) for a in (kf, std, real, norm))
    times = pd.DatetimeIndex(np.concatenate(times))
    print(f"{len(times):,} time steps x {len(ids)} sensors, {np.isfinite(real).mean():.0%} real readings")

    # crash windows: True while a detected crash was congesting its sensor
    in_crash = np.zeros(kf.shape, dtype=bool)
    col = {s: j for j, s in enumerate(ids)}
    for _, w in crash_windows().iterrows():
        if w["sensor"] not in col:
            continue
        rows = np.where((times >= w["start"]) & (times <= w["end"]))[0]
        in_crash[rows, col[w["sensor"]]] = True
    print(f"{in_crash.sum():,} sensor-steps inside crash windows")

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "gnn_data.npz")
    np.savez_compressed(out, speed_kf=kf, kf_std=std, real=real, normal=norm, in_crash=in_crash,
                        times=times.values.astype("datetime64[ns]"), stations=ids.astype("int64"),
                        slot=np.asarray(times.hour * 12 + times.minute // 5, dtype="int16"),
                        weekend=np.asarray(times.day_of_week >= 5))
    print(f"saved {out} ({os.path.getsize(out) / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
    
