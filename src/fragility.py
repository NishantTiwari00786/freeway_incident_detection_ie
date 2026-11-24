"""
fragility analysis (H3): which stretches of freeway take longest to recover after a comparable crash

Setup:
    segments: each freeway direction is cut into 2 mile pieces, each crash belongs to one segment
    outcome: how long the crash area stayed congested (congestion_min from label_incidents.py)
    controls: crash severity (injury or not), peak hours, weekend, rain
    model: mixed effects model, log(congestion) ~ controls + (a separate effect for each segment)
           segments with only a few crashes are pulled toward the average, so 2 bad days can't make
           a segment look like the worst one
    road features: lane drop (fewer lanes at the next sensor downstream) and interchange
                   (a sensor from another freeway within half a mile)

Script:
    1. loads the detected crashes with weather, and gives each one a segment and a severity
    2. fits the model and turns each segment's effect into a factor (1.5 = crashes last 50% longer than average)
       then checks that the segment differences are real (with vs without segments)
       and stable (Oct-Mar and Apr-Sep fit separately, do the same segments come out slow?)
    3. finds lane drops and interchanges for every segment
    4. fits the model again with the road features, to test if they explain the slow segments
    5. saves the ranking and a chart of the most fragile segments

Outputs:

data/processed/fragility.csv               one row per segment: crashes, fragility factor, road features
figures/fig7_fragility.png                 the most fragile segments

Run after load_weather.py
"""

import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import chi2, spearmanr

LABELS_FILE = "data/processed/crash_labels_weather.parquet"
STATIONS_FILE = "data/processed/graph/stations.parquet"
EDGES_FILE = "data/processed/graph/edges.parquet"
OUT_FILE = "data/processed/fragility.csv"
OUT_FIG = "figures/fig7_fragility.png"

SEGMENT_MI = 2.0        # length of each segment
INTERCHANGE_MI = 0.5    # another freeway's sensor this close = interchange (or the name says so, like '15/210 IC')
MIN_CRASHES = 5         # only rank segments with at least this many crashes
SPLIT_DATE = "2026-04-01"   # stability check: Oct-Mar vs Apr-Sep
CONTROLS = "log_min ~ C(severity) + peak + weekend + rain"

warnings.filterwarnings("ignore")   # statsmodels prints a lot of warnings while it tries different optimizers


def fit(model, reml=True):
    # the default optimizer often stops early on this data (and gives a too big segment variance)
    # powell is slower but finishes, so using it and checking it really converged
    r = model.fit(reml=reml, method="powell")
    if not r.converged:
        print("  WARNING: model did not converge")
    return r


def severity(t):
    # turning the CHP crash type into 3 groups
    t = str(t).lower()
    if "no inj" in t:
        return "no_injury"
    if "unkn" in t:
        return "unknown"
    if "inj" in t or "1141" in t or "fatal" in t:
        return "injury"                       # 1141 = ambulance sent
    return "unknown"


def miles_between(lat1, lon1, lat2, lon2):
    # distance in miles between two points on the map
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    h = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 3959 * 2 * np.arcsin(np.sqrt(h))


def road_features(stations, edges):
    # lane drop: a sensor whose next sensor downstream has fewer lanes
    lanes = stations.set_index("station")["lanes"]
    e = edges.copy()
    e["drop"] = e["upstream"].map(lanes) > e["downstream"].map(lanes)
    lane_drop = set(e.loc[e["drop"], "upstream"])

    # interchange: a sensor from another freeway within INTERCHANGE_MI
    s = stations.dropna(subset=["latitude", "longitude"])
    lat, lon, fwy = s["latitude"].to_numpy(), s["longitude"].to_numpy(), s["fwy"].to_numpy()
    d = miles_between(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
    other = fwy[:, None] != fwy[None, :]
    near_other = ((d <= INTERCHANGE_MI) & other).any(axis=1)
    # also by name, because freeways outside our 5 (like I-210) have no sensors in our data
    by_name = s["name"].astype(str).str.contains(r"\d{2,3}/\d{2,3}|\bIC\b|interch", case=False, regex=True)
    interchange = set(s["station"].to_numpy()[near_other | by_name.to_numpy()])
    return lane_drop, interchange


def segment_effects(c):
    # fits the model and returns each segment's effect (log scale)
    m = fit(smf.mixedlm(CONTROLS, c, groups=c["segment"]))
    return pd.Series({k: v.iloc[0] for k, v in m.random_effects.items()}, name="effect")


def main():
    os.makedirs(os.path.dirname(OUT_FIG), exist_ok=True)
    stations = pd.read_parquet(STATIONS_FILE)
    edges = pd.read_parquet(EDGES_FILE)

    # 1. detected crashes, each with a segment and a severity
    lab = pd.read_parquet(LABELS_FILE)
    c = lab[lab["labelable"] & lab["detected"].fillna(False).astype(bool) & lab["congestion_min"].notna()].copy()
    stations["segment"] = (stations["fwy"].astype(str) + stations["dir"] + " mi "
                           + (stations["abs_pm"] // SEGMENT_MI * SEGMENT_MI).astype(int).astype(str))
    c = c.merge(stations[["station", "segment"]].rename(columns={"station": "sensor"}), on="sensor", how="inner")
    c["severity"] = c["type"].map(severity)
    c["peak"] = c["peak"].astype(int)
    c["weekend"] = c["weekend"].astype(int)
    c["rain"] = c["rain"].fillna(False).astype(int)
    c["log_min"] = np.log(c["congestion_min"].clip(lower=5))   # crash durations are skewed, log makes them even
    print(f"{len(c):,} detected crashes in {c['segment'].nunique()} segments of {SEGMENT_MI:.0f} miles")
    print("severity:", c["severity"].value_counts().to_dict())

    # 2. mixed effects model: controls + a separate effect for each segment
    m = fit(smf.mixedlm(CONTROLS, c, groups=c["segment"]))
    effect = pd.Series({k: v.iloc[0] for k, v in m.random_effects.items()}, name="effect")
    print("\nControls (on log scale, so exp(x) = how many times longer):")
    for name in m.fe_params.index[1:]:
        print(f"  {name:30s} x{np.exp(m.fe_params[name]):.2f}   p = {m.pvalues[name]:.3f}")

    # 2b. are the segment differences real? same model with and without segments, compare how well they fit
    with_seg = fit(smf.mixedlm(CONTROLS, c, groups=c["segment"]), reml=False)   # reml=False so the two fits can be compared
    no_seg = smf.ols(CONTROLS, c).fit()
    lr = 2 * (with_seg.llf - no_seg.llf)
    p_seg = 0.5 * chi2.sf(lr, 1)          # half, because a variance can't go below 0
    sd = np.sqrt(m.cov_re.iloc[0, 0])
    print(f"\nSegments matter?  likelihood ratio = {lr:.1f}, p = {p_seg:.2g}")
    print(f"  typical segment is x{np.exp(-sd):.2f} to x{np.exp(sd):.2f} of average (1 std)")

    # 2c. is the ranking stable? fit each half of the year separately and compare
    first = c["timestamp"] < SPLIT_DATE
    e1, e2 = segment_effects(c[first]), segment_effects(c[~first])
    n1 = c[first].groupby("segment").size()
    n2 = c[~first].groupby("segment").size()
    both = n1[(n1 >= MIN_CRASHES)].index.intersection(n2[n2 >= MIN_CRASHES].index)
    rho, p_rho = spearmanr(e1[both], e2[both])
    top1 = set(e1[both].nlargest(10).index)
    top2 = set(e2[both].nlargest(10).index)
    print(f"\nStable across the year?  {len(both)} segments with {MIN_CRASHES}+ crashes in both halves")
    print(f"  rank correlation Oct-Mar vs Apr-Sep = {rho:.2f}, p = {p_rho:.2g}")
    print(f"  top 10 in both halves: {len(top1 & top2)} of 10  {sorted(top1 & top2)}")

    # 3. road features per segment
    lane_drop, interchange = road_features(stations, edges)
    seg = stations.groupby("segment").agg(
        lane_drop=("station", lambda s: s.isin(lane_drop).any()),
        interchange=("station", lambda s: s.isin(interchange).any()),
        example_name=("name", "first"),
        city=("city", "first"))
    counts = c.groupby("segment").agg(crashes=("congestion_min", "size"), median_min=("congestion_min", "median"))
    table = counts.join(effect).join(seg)
    table["factor"] = np.exp(table["effect"])                  # 1.5 = crashes here last 50% longer than average

    # 4. do lane drops and interchanges explain the slow segments?
    c = c.merge(seg[["lane_drop", "interchange"]].astype(int), left_on="segment", right_index=True, how="left")
    m2 = fit(smf.mixedlm(CONTROLS + " + lane_drop + interchange", c, groups=c["segment"]))
    print("\nRoad features (controls included):")
    for name in ("lane_drop", "interchange"):
        n_seg = int(seg.loc[seg.index.isin(c["segment"]), name].sum())
        print(f"  {name:12s} x{np.exp(m2.fe_params[name]):.2f} longer   p = {m2.pvalues[name]:.3f}   ({n_seg} segments)")

    # 5. ranking and chart
    ranked = table[table["crashes"] >= MIN_CRASHES].sort_values("factor", ascending=False)
    ranked.round(3).to_csv(OUT_FILE)
    print(f"\nMost fragile segments (at least {MIN_CRASHES} crashes):")
    print(ranked[["crashes", "median_min", "factor", "lane_drop", "interchange", "example_name"]].head(10).round(2).to_string())
    print(f"\nRange of segment factors: x{ranked['factor'].min():.2f} to x{ranked['factor'].max():.2f}")

    top = ranked.head(15)[::-1]
    label = [f"{s}  ({str(r.example_name)[:20]})" for s, r in top.iterrows()]
    colors = ["#c2410c" if (r.lane_drop or r.interchange) else "#256abf" for _, r in top.iterrows()]
    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=160)
    ax.barh(label, top["factor"], color=colors, height=0.65)
    ax.axvline(1, color="#1f1f1e", lw=1)
    for y, (f, n) in enumerate(zip(top["factor"], top["crashes"])):
        ax.text(f + 0.01, y, f"x{f:.2f} ({n} crashes)", va="center", fontsize=8, color="#6b6a66")
    ax.set_xlabel("Crash congestion compared with an average segment (controls included)", color="#6b6a66", fontsize=9)
    ax.set_title("The 15 most fragile 2-mile segments", loc="left", fontsize=11, color="#1f1f1e")
    ax.tick_params(labelsize=8, colors="#6b6a66")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_xlim(0.9, top["factor"].max() * 1.18)
    fig.text(0.01, -0.01, "Orange = has a lane drop or interchange. x1.0 = average segment.", fontsize=7.5, color="#6b6a66")
    fig.tight_layout()
    fig.savefig(OUT_FIG, bbox_inches="tight")
    print(f"saved {OUT_FIG}")


if __name__ == "__main__":
    main()