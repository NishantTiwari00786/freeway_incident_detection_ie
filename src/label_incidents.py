"""

Labeling each crash with how bad it is: how far the backup spread and how long it lasted 

For each unique crash (not on outage days) I looked at the sensors around it from 30 minutes before to 4 hours 
after the CHP report and compare actual speed normal speed for that sensor and time 


A sensor is "congested" in a 5 minute slot when its speed is at least DROP_MPH below normal
(and it is a real reading). From that we measure:
 
    detected        did the crash sensor (or the next one or two upstream) become congested near the report time
    onset           first congested slot near the crash, compared with the CHP report time (negative = sensors saw it first)
    congestion_min  how long the crash area stayed congested (onset to the first slot where it clears for 15 minutes)
                    (different from CHP's duration_min, which is when CHP closed the incident)
    in_existing_jam the area was already congested before the crash (e.g. rush hour), so the crash can't be separated
    queue_mi        how far (miles) the backup reached upstream from the first congested sensor at the crash
    min_speed       lowest speed at the crash sensors during the event
    peak / weekend  when the crash happened
 
Outputs:
 
data/processed/crash_labels.parquet   one row per labeled crash
 
Run after build_profile.py and load_incidents.py
    
    
    
"""

 
import os
 
import numpy as np
import pandas as pd
 
 
IN_DIR = "data/processed/5min"
INCIDENTS_FILE = "data/processed/incidents.parquet"
PROFILE_FILE = "data/processed/profile.parquet"
STATIONS_FILE = "data/processed/graph/stations.parquet"
OUT_FILE = "data/processed/crash_labels.parquet"
 
DROP_MPH = 20          # this much below normal counts as congested
BEFORE_MIN = 60        # look this far before the CHP report
AFTER_MIN = 360        # and this far after (severe crashes can last 4+ hours)
ONSET_WINDOW_MIN = 30  # the slowdown has to start within 30 min of the report to count as this crash
CLEAR_SLOTS = 3        # 3 slots (15 min) back to normal means the crash area has cleared
UPSTREAM_MI = 6        # how far upstream to look for the backup
MAX_SENSOR_MI = 1.0    # skip crashes more than 1 mile from any sensor
 
 
def upstream_order(stations):
    """For every (freeway, direction), the sensors sorted in travel order, with their postmiles."""
    order = {}
    for (fwy, d), g in stations.groupby(["fwy", "dir"]):
        g = g.sort_values("abs_pm", ascending=(d in ("N", "E")))   # travel order, same as build_graph.py
        order[(fwy, d)] = (g["station"].to_numpy(), g["abs_pm"].to_numpy())
    return order
 
 
def label_one(c, day_df, prof, order):
    ids, pms = order[(c["freeway"], c["direction"])]
    k = int(np.where(ids == c["sensor"])[0][0])
 
    # sensors upstream of the crash (traffic comes FROM them), within UPSTREAM_MI, plus the crash sensor
    up_idx = [i for i in range(k, -1, -1) if abs(pms[i] - pms[k]) <= UPSTREAM_MI]
    up_ids = ids[up_idx]
    up_dist = np.abs(pms[up_idx] - pms[k])
 
    t0 = c["timestamp"].floor("5min")
    w = day_df[(day_df["station"].isin(up_ids)) &
               (day_df["timestamp"] >= t0 - pd.Timedelta(minutes=BEFORE_MIN)) &
               (day_df["timestamp"] <= t0 + pd.Timedelta(minutes=AFTER_MIN))].copy()
    if w.empty:
        return None
    w["weekend"] = w["timestamp"].dt.dayofweek >= 5
    w["slot"] = (w["timestamp"].dt.hour * 12 + w["timestamp"].dt.minute // 5).astype("int16")
    w = w.merge(prof, on=["station", "weekend", "slot"], how="left")
    w["usable"] = (w["pct_observed"] > 0) & w["trusted"].fillna(False).astype(bool)
    w["congested"] = w["usable"] & (w["avg_speed"] <= w["median_speed"] - DROP_MPH)
 
    # speed x time grid: rows = time slots, columns = sensors ordered from the crash going upstream
    times = pd.date_range(t0 - pd.Timedelta(minutes=BEFORE_MIN), t0 + pd.Timedelta(minutes=AFTER_MIN), freq="5min")
    cong = w.pivot_table(index="timestamp", columns="station", values="congested", aggfunc="max")
    cong = cong.reindex(index=times, columns=up_ids).fillna(False).astype(bool)
    usable = w.pivot_table(index="timestamp", columns="station", values="usable", aggfunc="max")
    usable = usable.reindex(index=times, columns=up_ids).fillna(False).astype(bool)
 
    # the "crash area" = the crash sensor and the next 2 upstream sensors (the queue forms behind the crash)
    near = cong.iloc[:, :3].any(axis=1)
    near_usable = usable.iloc[:, :3].any(axis=1)
    if near_usable.mean() < 0.5:
        return {"labelable": False}
 
    # onset: the slot where the crash area goes from free flowing to congested, within ONSET_WINDOW_MIN of the report
    nv = near.to_numpy()
    win = (times >= t0 - pd.Timedelta(minutes=ONSET_WINDOW_MIN)) & (times <= t0 + pd.Timedelta(minutes=ONSET_WINDOW_MIN))
    first_in_win = int(np.argmax(win))
    # if the area was already congested right before the window, the crash happened inside an existing jam
    # (like rush hour), so we can't tell how much of the slowdown is the crash
    if nv[max(first_in_win - 3, 0):first_in_win].all():
        return {"labelable": True, "detected": False, "in_existing_jam": True}
    starts = [h for h in np.where(nv & win)[0] if h > 0 and not nv[h - 1]]
    out = {"labelable": True, "detected": len(starts) > 0, "in_existing_jam": False}
    if not out["detected"]:
        return out
    s = starts[0]
    out["onset_vs_report_min"] = (times[s] - t0).total_seconds() / 60
 
    # duration: from onset until the area stays clear for CLEAR_SLOTS slots in a row
    e = s
    while e < len(nv) and not (not nv[e:e + CLEAR_SLOTS].any() and e + CLEAR_SLOTS <= len(nv)):
        e += 1
    out["congestion_min"] = (e - s) * 5
    out["censored"] = e >= len(nv) - CLEAR_SLOTS      # still congested when the 6 h window ended
 
    # queue length: in each slot, start at the first congested sensor in the crash area and walk upstream.
    # keep going while sensors are congested, skip sensors with no real data (unknown, not proof the queue ended),
    # and stop at the first real sensor that is flowing normally. the queue is the farthest point reached.
    # (the closest sensor can be just DOWNSTREAM of the crash, where speeds go up, so we don't require it to be congested)
    queue = 0.0
    c_arr = cong.to_numpy()[s:max(e, s + 1)]
    u_arr = usable.to_numpy()[s:max(e, s + 1)]
    for row, urow in zip(c_arr, u_arr):
        start = next((i for i in range(min(3, len(row))) if row[i]), None)
        if start is None:
            continue
        far = start
        for j in range(start + 1, len(row)):
            if row[j]:
                far = j
            elif urow[j]:
                break
        queue = max(queue, up_dist[far] - up_dist[start])
    out["queue_mi"] = round(float(queue), 2)
 
    sp = w[w["station"].isin(up_ids[:3]) & w["usable"]]
    sp = sp[(sp["timestamp"] >= times[s]) & (sp["timestamp"] <= times[min(e, len(times) - 1)])]
    out["min_speed"] = float(sp["avg_speed"].min()) if len(sp) else np.nan
    return out
 
 
def main():
    inc = pd.read_parquet(INCIDENTS_FILE)
    prof = pd.read_parquet(PROFILE_FILE, columns=["station", "weekend", "slot", "median_speed", "trusted"])
    stations = pd.read_parquet(STATIONS_FILE)
    order = upstream_order(stations)
 
    crashes = inc[inc["is_crash"] & ~inc["is_duplicate"] & ~inc["on_outage_day"] &
                  (inc["sensor_dist_mi"] <= MAX_SENSOR_MI)].copy()
    crashes["day"] = crashes["timestamp"].dt.strftime("%Y_%m_%d")
    print(f"{len(crashes):,} crashes to label")
 
    rows = []
    for n, (day, group) in enumerate(crashes.groupby("day"), 1):
        path = os.path.join(IN_DIR, f"{day}.parquet")
        if not os.path.exists(path):
            continue
        day_df = pd.read_parquet(path, columns=["timestamp", "station", "pct_observed", "avg_speed"])
        for _, c in group.iterrows():
            r = label_one(c, day_df, prof, order)
            if r is not None:
                rows.append({"incident_id": c["incident_id"], **r})
        if n % 30 == 0:
            print(f"  {n} days done, {len(rows):,} crashes labeled so far")
 
    lab = pd.DataFrame(rows).merge(crashes.drop(columns="day"), on="incident_id")
    hour = lab["timestamp"].dt.hour
    lab["peak"] = hour.between(6, 8) | hour.between(15, 18)
    lab["weekend"] = lab["timestamp"].dt.dayofweek >= 5
    lab.to_parquet(OUT_FILE, index=False)
 
    ok = lab[lab["labelable"]]
    det = ok[ok["detected"]]
    print(f"\nCrashes with enough real sensor data to label: {len(ok):,} of {len(lab):,}")
    print(f"Showed a clear slowdown (>= {DROP_MPH} mph below normal near the report time): {len(det):,} ({len(det)/len(ok):.0%})")
    print(f"Sensors saw it before the CHP report: {(det['onset_vs_report_min'] < 0).mean():.0%} "
          f"| median onset vs report: {det['onset_vs_report_min'].median():+.0f} min")
    print(f"Duration (min): median {det['congestion_min'].median():.0f}, 90th pct {det['congestion_min'].quantile(.9):.0f}"
          f" | still congested after 6 h: {det['censored'].mean():.1%}")
    print(f"Queue (mi): median {det['queue_mi'].median():.1f}, 90th pct {det['queue_mi'].quantile(.9):.1f}")
    print(f"Crashes inside an existing jam (can't separate): {ok['in_existing_jam'].fillna(False).sum():,}")
    print(f"\nPeak vs off-peak median congestion: {det[det['peak']]['congestion_min'].median():.0f} vs {det[~det['peak']]['congestion_min'].median():.0f} min")
 
 
if __name__ == "__main__":
    main()
 