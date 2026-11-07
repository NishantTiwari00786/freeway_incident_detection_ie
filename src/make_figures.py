"""
Making the figures for Progress Report II from the processed data

Figures (saved to figures/):
    1. fig1_data_quality.png      share of sensors with real data each day, outage days marked
    2. fig2_crashes_by_freeway.png  unique crashes per freeway and direction
    3. fig3_crashes_by_hour.png     unique crashes by hour of day, weekday vs weekend
    4. fig4_crash_impact.png        how long detected crashes stayed congested, and how far the backup spread
    5. fig5_detection_timing.png    when sensors saw the slowdown compared with the CHP report

Run after load_incidents.py and label_incidents.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

OUT = "figures"
QUALITY_FILE = "data/processed/station_quality.parquet"
INCIDENTS_FILE = "data/processed/incidents.parquet"
LABELS_FILE = "data/processed/crash_labels.parquet"

# one quiet style for every figure
BLUE, LIGHT, GRAY, RED = "#256abf", "#9ec5f4", "#c9c8c3", "#c2410c"
INK, MUTED, GRID = "#1f1f1e", "#6b6a66", "#ecebe7"
plt.rcParams.update({"font.size": 9, "axes.titlesize": 11, "axes.titlelocation": "left",
                     "axes.edgecolor": "#d6d5d0", "axes.labelcolor": MUTED, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 160, "savefig.bbox": "tight"})


def save(fig, name):
    fig.savefig(os.path.join(OUT, name))
    plt.close(fig)
    print("saved", name)


# 1. data quality over the year
def fig_quality():
    q = pd.read_parquet(QUALITY_FILE)
    daily = q.groupby("date")["pct_observed"].apply(lambda s: (s > 0).mean() * 100)
    outages = daily[daily == 0]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(daily.index, daily.values, color=BLUE, lw=1.6)
    ax.scatter(outages.index, outages.values, color=RED, s=18, zorder=3, label=f"Full outage ({len(outages)} days)")
    ax.set_ylim(0, 100)
    ax.set_ylabel("% of sensors with real data")
    ax.set_title(f"About half of the sensors report real data on a typical day "
                 f"(median {daily[daily > 0].median():.0f}%)", color=INK)
    ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.legend(frameon=False, loc="lower left")
    save(fig, "fig1_data_quality.png")


# 2. crashes by freeway and direction
def fig_freeway(inc):
    u = inc[inc["is_crash"] & ~inc["is_duplicate"]]
    t = u.groupby(["freeway", "direction"]).size().reset_index(name="n")
    t["label"] = t.apply(lambda r: f"{'I' if r.freeway in (10, 15, 215) else 'SR'}-{r.freeway} {r.direction}", axis=1)
    t = t.sort_values("n")
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.barh(t["label"], t["n"], color=BLUE, height=0.6)
    for y, n in enumerate(t["n"]):
        ax.text(n + t["n"].max() * 0.01, y, f"{n:,}", va="center", color=MUTED, fontsize=8)
    ax.set_xlabel("Unique crashes, Oct 2025 to Sep 2026")
    ax.set_title(f"I-15 and I-10 have the most crashes ({len(u):,} unique crashes in total)", color=INK)
    ax.xaxis.grid(True, color=GRID); ax.set_axisbelow(True)
    save(fig, "fig2_crashes_by_freeway.png")


# 3. crashes by hour of day
def fig_hour(inc):
    u = inc[inc["is_crash"] & ~inc["is_duplicate"]].copy()
    u["hour"] = u["timestamp"].dt.hour
    u["weekend"] = u["timestamp"].dt.day_of_week >= 5
    # average per day, so weekdays (5) and weekends (2) are comparable
    n_days = u.groupby("weekend")["timestamp"].apply(lambda s: s.dt.date.nunique())
    t = u.groupby(["weekend", "hour"]).size().unstack(0).fillna(0) / n_days
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.plot(t.index, t[False], color=BLUE, lw=2, marker="o", ms=3, label="Weekday")
    ax.plot(t.index, t[True], color=GRAY, lw=2, marker="o", ms=3, label="Weekend")
    ax.axvspan(6, 9, color=GRID, zorder=0); ax.axvspan(15, 19, color=GRID, zorder=0)
    ax.set_xticks(range(0, 24, 3)); ax.set_xticklabels([f"{h}:00" for h in range(0, 24, 3)])
    ax.set_xlabel("Hour of day (shaded = peak hours)"); ax.set_ylabel("Crashes per day")
    ax.set_title("Crashes per day by hour, weekday vs weekend", color=INK)
    ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
    ax.legend(frameon=False)
    save(fig, "fig3_crashes_by_hour.png")


# 4. crash impact: congestion duration and backup length
def fig_impact(lab):
    d = lab[lab["labelable"] & lab["detected"].fillna(False).astype(bool)]
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.4))
    a.hist(d["congestion_min"].clip(upper=240), bins=range(0, 245, 10), color=BLUE, edgecolor="white")
    a.axvline(d["congestion_min"].median(), color=INK, ls="--", lw=1)
    a.text(d["congestion_min"].median() + 4, a.get_ylim()[1] * 0.9, f"median {d['congestion_min'].median():.0f} min", color=INK)
    a.set_xlabel("Minutes congested (240+ grouped at the end)"); a.set_ylabel("Crashes")
    a.set_title("How long the slowdown lasted", color=INK)
    b.hist(d["queue_mi"].clip(upper=6), bins=[x / 2 for x in range(0, 13)], color=BLUE, edgecolor="white")
    b.set_xlabel("Backup length upstream (miles, 6+ grouped)")
    b.set_title("How far the backup spread", color=INK)
    for ax in (a, b):
        ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
    fig.suptitle(f"{len(d):,} crashes with a clear slowdown", x=0.01, ha="left", color=MUTED, fontsize=9)
    fig.tight_layout()
    save(fig, "fig4_crash_impact.png")


# 5. detection timing vs CHP report
def fig_timing(lab):
    d = lab[lab["labelable"] & lab["detected"].fillna(False).astype(bool)]
    x = d["onset_vs_report_min"]
    before = (x < 0).mean()
    fig, ax = plt.subplots(figsize=(7, 3.4))
    bins = range(-30, 35, 5)
    counts = pd.cut(x, [b - 2.5 for b in bins] + [32.5], labels=list(bins)).value_counts().sort_index()
    ax.bar(counts.index.astype(int), counts.values, width=4, color=[LIGHT if b >= 0 else BLUE for b in counts.index.astype(int)])
    ax.axvline(0, color=INK, lw=1)
    top = counts.max() * 1.2
    ax.set_ylim(0, top)
    ax.text(-1.5, top * 0.95, "Sensors first", color=BLUE, fontweight="bold", ha="right", va="top")
    ax.text(1.5, top * 0.95, "CHP report first", color=MUTED, fontweight="bold", ha="left", va="top")
    ax.set_xlabel("Slowdown start minus CHP report time (minutes)"); ax.set_ylabel("Crashes")
    ax.set_title(f"Sensors saw the slowdown before CHP in {before:.0%} of detected crashes", color=INK)
    ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
    save(fig, "fig5_detection_timing.png")


def main():
    os.makedirs(OUT, exist_ok=True)
    inc = pd.read_parquet(INCIDENTS_FILE)
    lab = pd.read_parquet(LABELS_FILE)
    fig_quality()
    fig_freeway(inc)
    fig_hour(inc)
    fig_impact(lab)
    fig_timing(lab)


if __name__ == "__main__":
    main()