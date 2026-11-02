"""

Loading one year of CHP incident recors and match each incident to the sensor network 

PeMS publishes CHP incidents as one statewide per month

In this script: 
                1. we loop over all the files on monthy basis, and keep the district 8 incidents on our five freeways
                2. pulls the incident type out of the description 
                3. matches each incident to nearest sensor on the same freeway and direction (by postmile)
                4. flags likely duplicates: one incident is reported multiple times 
                5. flags incidents that happened on days when pems had no real sensor data at all 
                
output file = data/processed/incidents.parquet : one row per CHP record, with the matched sensor and flags 



"""
    

import glob
import os 

import numpy as np 
import pandas as pd 

CHP_DIR = "data/raw/chp"
STATIONS_FILE = "data/processed/graph/stations.parquet"
QUALITY_FILE = "data/processed/station_quality.parquet"
OUT_FILE = "data/processed/incidents.parquet"
DISTRICT = 8
FREEWAYS = [10, 15, 215, 60, 91]
 
# if two records are on the same freeway and direction and within 2 miles and 30 minutes of each other,
# we treat them as the same incident
DUP_MILES = 2.0
DUP_MINUTES = 30
 
# columns in the raw CHP file (it has no header row), order is from the pems clearinghouse field specification
NAMES = ["incident_id", "cc_code", "incident_number", "timestamp", "description", "location",
         "area", "zoom_map", "tb_xy", "latitude", "longitude", "district", "county_fips",
         "city_fips", "freeway", "direction", "state_pm", "abs_pm", "severity", "duration_min"]
 
# columns we are keeping for the analysis
KEEP = ["incident_id", "timestamp", "type", "description", "location", "freeway", "direction",
        "abs_pm", "latitude", "longitude", "duration_min"]
 
 
# loading all the CHP files and keeping only our study network
def load_chp():
    files = sorted(glob.glob(os.path.join(CHP_DIR, "all_text_chp_incidents_month_*")))
    print(f"{len(files)} monthly CHP files found")
    df = pd.concat((pd.read_csv(f, header=None, names=NAMES) for f in files), ignore_index=True)
    print(f"{len(df):,} incidents statewide")
 
    df = df[(df["district"] == DISTRICT) & df["freeway"].isin(FREEWAYS)].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%m/%d/%Y %H:%M:%S")
 
    # "1183-Trfc Collision-Unkn Inj" -> "Trfc Collision-Unkn Inj", the number before the first dash is the CHP radio code
    df["type"] = df["description"].str.split("-", n=1).str[1].str.strip()
 
    # sorting by time because the duplicate check needs the records in order
    return df[KEEP].sort_values("timestamp").reset_index(drop=True)
 
 
# matching each incident to the closest sensor on the same freeway and direction
def match_to_sensors(inc, stations):
    inc["sensor"] = pd.NA
    inc["sensor_dist_mi"] = np.nan
 
    for (fwy, d), idx in inc.groupby(["freeway", "direction"]).groups.items():
        s = stations[(stations["fwy"] == fwy) & (stations["dir"] == d)].sort_values("abs_pm")
        if s.empty:
            continue
        pm = s["abs_pm"].to_numpy()
        x = inc.loc[idx, "abs_pm"].to_numpy()
 
        # searchsorted tells us where each incident would fit in the sorted sensor list,
        # so the closest sensor is either the one right before or right after that spot
        right = np.clip(np.searchsorted(pm, x), 1, len(pm) - 1)
        left = right - 1
        nearest = np.where(np.abs(pm[right] - x) < np.abs(pm[left] - x), right, left)
 
        inc.loc[idx, "sensor"] = s["station"].to_numpy()[nearest]
        inc.loc[idx, "sensor_dist_mi"] = np.abs(pm[nearest] - x)
 
    inc["sensor"] = inc["sensor"].astype("Int64")
    return inc
 
 
# flagging duplicate reports of the same incident
# a record is a duplicate if an earlier record on the same freeway and direction is within DUP_MILES and DUP_MINUTES,
# the first report is kept as the original
def flag_duplicates(inc):
    dup = np.zeros(len(inc), dtype=bool)
 
    for _, g in inc.groupby(["freeway", "direction"]):
        t = g["timestamp"].to_numpy()
        pm = g["abs_pm"].to_numpy()
        pos = g.index.to_numpy()
 
        for k in range(1, len(g)):
            j = k - 1
            # going back through the earlier records as long as they are inside the time window
            while j >= 0 and (t[k] - t[j]) <= np.timedelta64(DUP_MINUTES, "m"):
                if abs(pm[k] - pm[j]) <= DUP_MILES:
                    dup[pos[k]] = True
                    break
                j -= 1
 
    inc["is_duplicate"] = dup
    return inc
 
 
def main():
    inc = load_chp()
    stations = pd.read_parquet(STATIONS_FILE)
 
    inc = match_to_sensors(inc, stations)
    inc = flag_duplicates(inc)
 
    # crashes are collisions and hit and runs, the rest are hazards, fires, animals, etc.
    inc["is_crash"] = inc["type"].str.contains("Collision|Hit and Run", na=False)
 
    # checking which days pems had no real data at all (every sensor imputed),
    # we can't measure incidents on these days so we flag them
    q = pd.read_parquet(QUALITY_FILE)
    daily_real = q.groupby("date")["pct_observed"].apply(lambda s: (s > 0).sum())
    outage_days = set(daily_real[daily_real == 0].index.date)
    inc["on_outage_day"] = inc["timestamp"].dt.date.isin(outage_days)
 
    inc.to_parquet(OUT_FILE, index=False)
 
    # summary, counting only unique incidents (no duplicates)
    u = inc[~inc["is_duplicate"]]
    print(f"\n{len(inc):,} incidents on our freeways, {inc['timestamp'].min():%Y-%m-%d} to {inc['timestamp'].max():%Y-%m-%d}")
    print(f"Likely duplicates: {inc['is_duplicate'].sum():,} ({inc['is_duplicate'].mean():.0%})")
    print(f"Unique incidents: {len(u):,}  |  unique crashes: {u['is_crash'].sum():,}")
    print(f"Within 0.5 mi of a sensor: {(u['sensor_dist_mi'] <= 0.5).mean():.0%}  |  within 1 mi: {(u['sensor_dist_mi'] <= 1).mean():.0%}")
    print(f"On PeMS outage days ({len(outage_days)} days): {u['on_outage_day'].sum():,} unique incidents")
    print("\nTop incident types (unique):")
    print(u["type"].value_counts().head(8).to_string())
    print("\nUnique crashes by freeway and direction:")
    print(u[u["is_crash"]].groupby(["freeway", "direction"]).size().unstack(fill_value=0))
 
 
if __name__ == "__main__":
    main()
