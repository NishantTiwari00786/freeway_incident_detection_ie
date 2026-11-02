"""
Building the sensor graph for the study network from all pems metadata files

pems metadata files are published by Caltrans and contain the sensor location, freeway, direction, lane type, and other information,
and a new version is published whenever sensors are added or removed

In our timeframe, we have 25 versions

Script:
    1. reads every version and tags each row with version date
    2. keeps only the Mainline sensors
    3. Keeps one row per sensor, and records the first and last version date for each sensor
    4. orders sensors along each freeway in the direction that traffic flows, and links each sensor to the next one downstream

Outputs:

data/processed/graph/stations.parquet one row per sensor (location, lanes, first/last seen)
data/processed/graph/edges.parquet   one row per upstream --> downstream

"""

import glob
import os

import pandas as pd


META_DIR = "data/raw/meta"
OUT_DIR = "data/processed/graph"

FREEWAYS = [10, 15, 215, 60, 91]

LONG_GAP_MI = 2.0 # links longer mean it is either a rural area or missing sensor problem



# reading the metadata file
def read_meta(path):
    header = open(path).readline().rstrip("\n").split("\t")
    m = pd.read_csv(path, sep="\t", skiprows=1, names=header + ["_extra"], index_col=False)
    m = m.drop(columns="_extra")
    m.columns = [c.strip().lower() for c in m.columns] # 'Abs_PM' -> 'abs_pm'
    # d08_text_meta_2026_01_08.txt -> 2026-01-08
    m["version"] = pd.Timestamp(os.path.basename(path)[14:24].replace("_", "-"))
    return m

def main():

    files = sorted(glob.glob(os.path.join(META_DIR, "d08_text_meta_*.txt")))
    print(f"{len(files)} metadata versions found")

    meta = pd.concat([read_meta(f) for f in files], ignore_index=True)
    meta = meta[(meta["type"] == "ML") & meta["fwy"].isin(FREEWAYS)] # keep only mainline sensors on the freeways we care about

    # checking when each sensor was in network

    seen = meta.groupby("id")["version"].agg(first_seen="min", last_seen="max")

    # checking if any sensor moved within versions

    moved = meta.groupby("id")["abs_pm"].nunique()
    print(f"Sensors that changed postmile between versions: {(moved > 1).sum()}")

    # One row per sensor: its most recent version
    stations = (meta.sort_values("version")
                    .drop_duplicates("id", keep="last")
                    .set_index("id")
                    .join(seen)
                    .reset_index())
    stations = stations[["id", "fwy", "dir", "abs_pm", "latitude", "longitude",
                         "lanes", "name", "city", "first_seen", "last_seen"]]
    stations = stations.rename(columns={"id": "station"})

    # linking each sensor to next one downstream

    edges = []

    for (fwy, d), g in stations.groupby(["fwy", "dir"]):
        g = g.sort_values("abs_pm", ascending=(d in ("N", "E")))   # now in travel order
        ids, pms = g["station"].tolist(), g["abs_pm"].tolist()
        for up, down, pm_up, pm_down in zip(ids, ids[1:], pms, pms[1:]):
            edges.append({"upstream": up, "downstream": down, "fwy": fwy, "dir": d,
                          "gap_mi": round(abs(pm_down - pm_up), 3)})
    edges = pd.DataFrame(edges)

    os.makedirs(OUT_DIR, exist_ok=True)
    stations.to_parquet(os.path.join(OUT_DIR, "stations.parquet"), index=False)
    edges.to_parquet(os.path.join(OUT_DIR, "edges.parquet"), index=False)

    print(f"\n{len(stations)} sensors, {len(edges)} links")
    print(stations.groupby(["fwy", "dir"]).size().unstack(fill_value=0), "\n")
    print(f"Median gap between neighbors: {edges['gap_mi'].median():.2f} mi")
    print(f"Links longer than {LONG_GAP_MI} mi: {(edges['gap_mi'] > LONG_GAP_MI).sum()}")
    print(f"Co-located sensors (gap = 0): {(edges['gap_mi'] == 0).sum()}")


if __name__ == "__main__":
    main()