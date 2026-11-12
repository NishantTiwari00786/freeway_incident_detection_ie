"""
    Loading NOAA weather and comparing crashes in rain vs dry conditions
    
    I used NOAA local climatological data (lcd) from airport stations around the inland empire, 
    one file per station per year (2025 & 2026). Each sensor gets the weather data from nearest station 
    
    
    Script: 
    1. reads all LCD files and keeps hourly weather columns (similar to what I did in CS252 weather)
    2. cleans NOAA codes: T = trace of rain, M = missing and any other flags 
    3. Builds one row per station per hour and marks the hour as rain if any precipitation fell 
        when dry lcd leaves it blank meaning no rain 
    4. matches every crash to nearest station and the hour it happened
    5. compare rain vs dry: how often crashes happen , how often they cause slowdown, how long the congestion lasts, and how far is the backup 
    
    
Outputs: 
data/processed/weather_hourly.parquet one row per station per hour
data/processed/crash_labels_weather.parquet crash labels with rain/ weather columns added 

this file works after running label_incidents.py
"""

import glob 
import os

import numpy as np 
import pandas as pd 

WEATHER_DIR = 'data/raw/weather'
LABELS_FILE = 'data/processed/crash_labels.parquet'
INCIDENTS_FILE = 'data/processed/incidents.parquet'
OUT_HOURLY = 'data/processed/weather_hourly.parquet'
OUT_LABELS = 'data/processed/crash_labels_weather.parquet'

START, END = "2025-10-01", "2026-10-01"

# the hourly column LCD files 

HOURLY = {
    "HourlyPrecipitation": "precip",
    "HourlyVisibility": "visibility",
    "HourlyDryBulbTemperature": "temp",
    "HourlyWindSpeed": "wind",

}

TRACE = 0.001 # trace of rain is too small to measure, so road wet will be considered rain in my analysis

def clean(col):
    # NOAA mixes codes and numbers in same column, so cleaning them is essential
    s = col.astype(str).str.strip()
    s = s.replace({"T": str(TRACE), "M": "", "nan": "", "None": ""})
    s = s.str.replace(r"[A-Za-z*]+$", "", regex=True) # 0.25s --> 0.25
    return pd.to_numeric(s, errors="coerce")


def load_station_files():
    files = sorted(glob.glob(os.path.join(WEATHER_DIR, "LCD_*.csv")))
    print(f"{len(files)} LCD files found")
    rows, stations = [], {}
    for f in files:
        w = pd.read_csv(f, low_memory=False)
        sid = os.path.basename(f).split("_")[1]                 # LCD_USW00003171_2025.csv -> USW00003171
        stations[sid] = (float(w["LATITUDE"].iloc[0]), float(w["LONGITUDE"].iloc[0]), str(w["NAME"].iloc[0]) if "NAME" in w else sid)
        keep = ["DATE"] + [c for c in HOURLY if c in w.columns]
        w = w[keep].rename(columns=HOURLY)
        w["DATE"] = pd.to_datetime(w["DATE"], errors="coerce")
        for c in HOURLY.values():
            if c in w.columns:
                w[c] = clean(w[c])
        w["station_id"] = sid
        rows.append(w.dropna(subset=["DATE"]))
    w = pd.concat(rows, ignore_index=True)
 
    # LCD has routine hourly reports plus extra special reports, so one hour can have several rows.
    # we take the max precipitation in the hour (special reports are partial totals, adding them would double count)
    w["hour"] = w["DATE"].dt.floor("h")
    hourly = (w.groupby(["station_id", "hour"])
                .agg(precip=("precip", "max"), visibility=("visibility", "min"),
                     temp=("temp", "mean"), wind=("wind", "mean"))
                .reset_index())
    hourly = hourly[(hourly["hour"] >= START) & (hourly["hour"] < END)]
 
    # LCD leaves precipitation BLANK when no rain fell (it does not write 0), so blank means dry.
    # we also build a full grid of every hour for every station, so hours with no report at all count as dry too
    grid = pd.MultiIndex.from_product([sorted(hourly["station_id"].unique()),
                                       pd.date_range(START, END, freq="h", inclusive="left")],
                                      names=["station_id", "hour"])
    hourly = hourly.set_index(["station_id", "hour"]).reindex(grid).reset_index()
    hourly["reported"] = hourly["temp"].notna() | hourly["precip"].notna()
    hourly["precip"] = hourly["precip"].fillna(0)
    hourly["rain"] = hourly["precip"] > 0
    return hourly, stations
 
 
def nearest_station(lat, lon, stations):
    # straight line distance is fine here because all stations are within about 100 miles of each other
    ids = list(stations)
    d = [np.hypot(lat - stations[s][0], (lon - stations[s][1]) * np.cos(np.radians(lat))) for s in ids]
    return ids[int(np.argmin(d))]
 
 
def main():
    hourly, stations = load_station_files()
    hourly.to_parquet(OUT_HOURLY, index=False)
    print("\nStations:")
    for sid, (lat, lon, name) in stations.items():
        h = hourly[hourly["station_id"] == sid]
        print(f"  {sid} {name[:35]:35s} hours: {len(h):,} | rain hours: {h['rain'].sum():,} ({h['rain'].mean():.1%})"
              f" | hours with a report: {h['reported'].mean():.0%}")
 
    lab = pd.read_parquet(LABELS_FILE)
    lab["station_id"] = [nearest_station(a, b, stations) for a, b in zip(lab["latitude"], lab["longitude"])]
    lab["hour"] = lab["timestamp"].dt.floor("h")
    lab = lab.merge(hourly[["station_id", "hour", "precip", "rain", "visibility"]], on=["station_id", "hour"], how="left")
    lab.to_parquet(OUT_LABELS, index=False)
 
    # 1) crash frequency: crashes per hour when it is raining vs dry, at each crash's nearest station
    inc = pd.read_parquet(INCIDENTS_FILE)
    crashes = inc[inc["is_crash"] & ~inc["is_duplicate"]].dropna(subset=["latitude", "longitude"]).copy()
    crashes["station_id"] = [nearest_station(a, b, stations) for a, b in zip(crashes["latitude"], crashes["longitude"])]
    crashes["hour"] = crashes["timestamp"].dt.floor("h")
    crashes = crashes.merge(hourly[["station_id", "hour", "rain"]], on=["station_id", "hour"], how="left")
    known = hourly
    rate = {}
    for r in (True, False):
        n_crash = (crashes["rain"] == r).sum()
        n_hours = (known["rain"] == r).sum()
        rate[r] = n_crash / n_hours
    print(f"\nCrashes per station-hour: rain {rate[True]:.3f} vs dry {rate[False]:.3f}"
          f"  ->  {rate[True] / rate[False]:.1f}x more crashes per hour when it rains")
 
    # 2) crash impact: rain vs dry, using only crashes we could measure
    ok = lab[lab["labelable"] & lab["rain"].notna()]
    print(f"\nMeasured crashes with weather: {len(ok):,} (rain: {(ok['rain'] == True).sum():,}, dry: {(ok['rain'] == False).sum():,})")
    print(f"{'':28s}{'rain':>10s}{'dry':>10s}")
    det = ok[ok["detected"].fillna(False).astype(bool)]
    def stat(r, f):
        return f(ok[ok["rain"] == r], det[det["rain"] == r])
    lines = [
        ("Caused a clear slowdown", lambda a, d: f"{len(d) / len(a):.0%}" if len(a) else "-"),
        ("Median congestion (min)", lambda a, d: f"{d['congestion_min'].median():.0f}" if len(d) else "-"),
        ("90th pct congestion (min)", lambda a, d: f"{d['congestion_min'].quantile(.9):.0f}" if len(d) else "-"),
        ("90th pct backup (mi)", lambda a, d: f"{d['queue_mi'].quantile(.9):.1f}" if len(d) else "-"),
        ("Detected crashes", lambda a, d: f"{len(d):,}"),
    ]
    for name, f in lines:
        print(f"{name:28s}{stat(True, f):>10s}{stat(False, f):>10s}")
 
 
if __name__ == "__main__":
    main()
 
    