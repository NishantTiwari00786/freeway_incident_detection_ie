"""

Interchange links: connecting freeways to each other where they cross

build_graph gives 10 separate chains (one per freeway direction), so a slowdown on SR-60 can 
never I-15 even if they intersect 

How: 
1. for every pair of freeway, we find sensor pairs closer than MAX_MI (sensors that intersect within a 1.0 mile)
3. keep only the closest pair for each crossing: take the closest pair, drop every other pair within same_crossing 
since links are two way

Outputs: 

data/processed/graph/interchange_edges.parquet     one row per link (sensor_a, sensor_b, distance)

edges.parquet is not changed, so kalman, gbm and labels are not affected

Run after build_graph.py

"""

import numpy as np 
import pandas as pd 

STATIONS_FILE = "data/processed/graph/stations.parquet"
OUT_FILE = "data/processed/graph/interchange_edges.parquet"

MAX_MI = 1.0 # two sensor on diff freeways closer than this = crossing
SAME_CROSSING_MI = 3.0 # other closer pairs within this distance belong to same crossing 

def miles_between(lat1, lon1, lat2, lon2):
    # distance in miles between two points 
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    h = np.sin((lat2-lat1)/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin((lon2-lon1)/2)**2 
    return 3959 * 2 * np.arcsin(np.sqrt(h))


def main():
    st = pd.read_parquet(STATIONS_FILE).dropna(subset=["latitude", "longitude"])
    st["route"] = st["fwy"].astype(str) + st["dir"]
    routes = sorted(st["route"].unique())

    links = []
    for i, ra in enumerate(routes):
        for rb in routes[i + 1:]:
            a = st[st["route"] == ra]
            b = st[st["route"] == rb]
            if a["fwy"].iloc[0] == b["fwy"].iloc[0]:
                continue                                   # 15N and 15S are the same freeway, not a crossing

            # 1. every sensor in a against every sensor in b
            d = miles_between(a["latitude"].to_numpy()[:, None], a["longitude"].to_numpy()[:, None],
                              b["latitude"].to_numpy()[None, :], b["longitude"].to_numpy()[None, :])
            ia, ib = np.where(d <= MAX_MI)
            pairs = sorted(zip(d[ia, ib], ia, ib))         # closest first

            # 2. closest pair = the crossing, then skip the other pairs near it
            chosen = []
            for dist, x, y in pairs:
                pa = a.iloc[x]
                near = any(miles_between(pa["latitude"], pa["longitude"], c["latitude"], c["longitude"]) < SAME_CROSSING_MI
                           for c in chosen)
                if not near:
                    chosen.append(pa)
                    pb = b.iloc[y]
                    links.append({"sensor_a": pa["station"], "sensor_b": pb["station"],
                                  "route_a": ra, "route_b": rb, "dist_mi": round(dist, 3),
                                  "name_a": pa["name"], "name_b": pb["name"],
                                  "lat": pa["latitude"], "lon": pa["longitude"]})

    links = pd.DataFrame(links)
    links.drop(columns=["name_a", "name_b"]).to_parquet(OUT_FILE, index=False)

    # 3. printing the crossings so we can check them against the real map
    links["fwys"] = links.apply(lambda r: " / ".join(sorted({r["route_a"][:-1], r["route_b"][:-1]}, key=int)), axis=1)
    print(f"{len(links)} interchange links\n")
    for fwys, g in links.groupby("fwys"):
        print(f"{fwys}: {len(g)} links")
        for _, r in g.sort_values("lat").iterrows():
            print(f"   {r['route_a']:>4} {str(r['name_a'])[:22]:22s} <-> {r['route_b']:>4} {str(r['name_b'])[:22]:22s}"
                  f"  {r['dist_mi']:.2f} mi   ({r['lat']:.3f}, {r['lon']:.3f})")
    print(f"\nsaved {OUT_FILE}")


if __name__ == "__main__":
    main()