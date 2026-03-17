"""
2025 Colorado Rockies – Team-Level Statcast Analysis
=====================================================
Metrics covered:
  1. Team sprint speed rankings (all 30 MLB teams)
  2. Pitcher 4-seam fastball (FF only) velocity — raw + home/road split
  3. Pitcher GB%, FB%, LD%, K% — overall + home/road delta
  4. Hitter exit velocity — overall + home/road delta
  5. Hitter launch angle — overall + home/road delta
  6. Hitter GB%, FB%, LD%, K% — overall + home/road delta

Requirements:
    pip install pybaseball pandas numpy matplotlib seaborn scipy

Data source: MLB Statcast via pybaseball (Baseball Savant)
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

import pybaseball
from pybaseball import statcast, statcast_sprint_speed

pybaseball.cache.enable()

# ── Season parameters ────────────────────────────────────────────────────────
SEASON_START = "2025-03-27"
SEASON_END   = "2025-09-28"
COL_ABBR     = "COL"

# ── Plotting style ────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)
COL_COLOR  = "#33006F"
GRAY_COLOR = "#90a0b0"


# ─────────────────────────────────────────────────────────────────────────────
# UTILITY HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def team_colors(teams):
    return [COL_COLOR if t == COL_ABBR else GRAY_COLOR for t in teams]

def rank_label(rank, total=30):
    return f"#{int(rank)} of {total}"

def add_tight_xaxis(ax, series, n_ticks=10, decimals=1, pad_pct=0.03):
    """
    Zoom x-axis into the  data range and add granular tick labels.
    """
    lo = series.min()
    hi = series.max()
    span = hi - lo if hi != lo else 1.0
    pad  = span * pad_pct
    x_lo = lo - pad
    x_hi = hi + span * 0.07          # a little right-margin for readability
    ax.set_xlim(x_lo, x_hi)
    ticks = np.linspace(x_lo, x_hi, n_ticks)
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter(f"%.{decimals}f"))
    ax.tick_params(axis="x", labelsize=8, rotation=30)


# ─────────────────────────────────────────────────────────────────────────────
# 1. SPRINT SPEED
# ─────────────────────────────────────────────────────────────────────────────

def analyze_sprint_speed():
    print("\n" + "="*60)
    print("1. SPRINT SPEED")
    print("="*60)

    speed_df = statcast_sprint_speed(2025)

    if speed_df is None or speed_df.empty:
        print("  ⚠  Sprint speed data not available yet for 2025.")
        return None

    team_speed = (
        speed_df
        .groupby("team")["sprint_speed"]
        .agg(avg_sprint_speed="mean", n_players="count")
        .reset_index()
        .sort_values("avg_sprint_speed", ascending=False)
        .reset_index(drop=True)
    )
    team_speed["rank"] = range(1, len(team_speed) + 1)

    col_row = team_speed[team_speed["team"] == COL_ABBR].iloc[0]
    print(f"\n  COL sprint speed: {col_row['avg_sprint_speed']:.2f} ft/sec  "
          f"({rank_label(col_row['rank'])})")
    print("\n  Top 5 fastest teams:")
    print(team_speed.head(5)[["rank","team","avg_sprint_speed"]].to_string(index=False))

    fig, ax = plt.subplots(figsize=(10, 8))
    sorted_df = team_speed.sort_values("avg_sprint_speed")
    ax.barh(sorted_df["team"], sorted_df["avg_sprint_speed"],
            color=team_colors(sorted_df["team"]))
    mlb_avg = team_speed["avg_sprint_speed"].mean()
    ax.axvline(mlb_avg, color="black", linestyle="--", linewidth=1,
               label=f"MLB avg: {mlb_avg:.2f}")
    ax.set_xlabel("Avg Sprint Speed (ft/sec)", fontweight="bold")
    ax.set_title("2025 MLB Team Sprint Speed Rankings", fontweight="bold")
    add_tight_xaxis(ax, sorted_df["avg_sprint_speed"], n_ticks=10, decimals=2)
    ax.legend()
    plt.tight_layout()
    plt.savefig("sprint_speed_rankings.png", dpi=150)
    plt.show()
    print("  → saved sprint_speed_rankings.png")
    return team_speed


# ─────────────────────────────────────────────────────────────────────────────
# LOAD STATCAST DATA
# ─────────────────────────────────────────────────────────────────────────────

def load_statcast_data():
    print("\n" + "="*60)
    print("LOADING STATCAST DATA (this may take several minutes)...")
    print("="*60)

    chunks = [
        ("2025-03-27", "2025-04-30"),
        ("2025-05-01", "2025-05-31"),
        ("2025-06-01", "2025-06-30"),
        ("2025-07-01", "2025-07-31"),
        ("2025-08-01", "2025-08-31"),
        ("2025-09-01", "2025-09-28"),
    ]

    dfs = []
    for start, end in chunks:
        print(f"  Pulling {start} → {end} …", end=" ", flush=True)
        try:
            df = statcast(start_dt=start, end_dt=end)
            dfs.append(df)
            print(f"{len(df):,} rows")
        except Exception as e:
            print(f"ERROR: {e}")

    if not dfs:
        raise RuntimeError("No Statcast data loaded.")

    data = pd.concat(dfs, ignore_index=True)
    print(f"\n  Total rows loaded: {len(data):,}")

    # Top of inning → away bats, home pitches; Bot → home bats, away pitches
    data["pitcher_is_home"] = (data["inning_topbot"] == "Top")
    data["batter_is_home"]  = (data["inning_topbot"] == "Bot")
    data["pitcher_team"]    = np.where(data["pitcher_is_home"], data["home_team"], data["away_team"])
    data["batter_team"]     = np.where(data["batter_is_home"],  data["home_team"], data["away_team"])

    return data


# ─────────────────────────────────────────────────────────────────────────────
# 2. 4-SEAM FASTBALL VELOCITY (FF only)
# ─────────────────────────────────────────────────────────────────────────────

def analyze_fastball_velocity(data):
    print("\n" + "="*60)
    print("2. 4-SEAM FASTBALL VELOCITY — FF only (home vs. road)")
    print("="*60)

    fb = data[data["pitch_type"] == "FF"].dropna(subset=["release_speed"]).copy()
    print(f"\n  Sample: {len(fb):,} four-seam fastballs from {fb['pitcher'].nunique()} pitchers")

    # Overall
    overall = (
        fb.groupby("pitcher_team")["release_speed"]
        .mean().reset_index()
        .rename(columns={"release_speed": "avg_velo"})
        .sort_values("avg_velo", ascending=False).reset_index(drop=True)
    )
    overall["rank"] = range(1, len(overall) + 1)

    col_ov = overall[overall["pitcher_team"] == COL_ABBR].iloc[0]
    print(f"\n  COL overall avg 4-seam: {col_ov['avg_velo']:.2f} mph  "
          f"({rank_label(col_ov['rank'])})")

    # Home / Road split
    split = (
        fb.groupby(["pitcher_team", "pitcher_is_home"])["release_speed"]
        .mean().unstack()
        .rename(columns={True: "home_velo", False: "road_velo"})
        .reset_index()
    )
    split["home_road_delta"] = split["home_velo"] - split["road_velo"]
    mlb_delta = split["home_road_delta"].mean()

    col_split = split[split["pitcher_team"] == COL_ABBR].iloc[0]
    print(f"\n  COL home velo:            {col_split['home_velo']:.2f} mph")
    print(f"  COL road velo:            {col_split['road_velo']:.2f} mph")
    print(f"  COL Δ (H−R):             {col_split['home_road_delta']:+.2f} mph")
    print(f"  MLB avg Δ (H−R):          {mlb_delta:+.2f} mph")
    print(f"  COL excess Coors effect:  {col_split['home_road_delta'] - mlb_delta:+.2f} mph")

    road_rank = (
        split[["pitcher_team","road_velo"]]
        .sort_values("road_velo", ascending=False).reset_index(drop=True)
    )
    road_rank["rank"] = range(1, len(road_rank)+1)
    col_road = road_rank[road_rank["pitcher_team"]==COL_ABBR].iloc[0]
    print(f"\n  COL road-adjusted rank: {rank_label(col_road['rank'])}  "
          f"({col_road['road_velo']:.2f} mph)")
    print("\n  Top 5 by road velo:")
    print(road_rank.head(5).to_string(index=False))

    # Plots
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    ov = overall.sort_values("avg_velo")
    axes[0].barh(ov["pitcher_team"], ov["avg_velo"], color=team_colors(ov["pitcher_team"]))
    mlb_ov = overall["avg_velo"].mean()
    axes[0].axvline(mlb_ov, color="black", linestyle="--", linewidth=1,
                    label=f"MLB avg: {mlb_ov:.2f}")
    axes[0].set_xlabel("Avg 4-Seam Velo (mph)", fontweight="bold")
    axes[0].set_title("Overall Avg 4-Seam Fastball Velocity", fontweight="bold")
    add_tight_xaxis(axes[0], ov["avg_velo"], n_ticks=10, decimals=1)
    axes[0].legend(fontsize=8)

    sp = split.sort_values("home_road_delta")
    axes[1].barh(sp["pitcher_team"], sp["home_road_delta"], color=team_colors(sp["pitcher_team"]))
    axes[1].axvline(0, color="black", linewidth=0.8)
    axes[1].axvline(mlb_delta, color="red", linestyle="--", linewidth=1,
                    label=f"MLB avg Δ: {mlb_delta:+.2f}")
    axes[1].set_xlabel("4-Seam Velo Δ Home − Road (mph)", fontweight="bold")
    axes[1].set_title("Home vs. Road 4-Seam Velo Delta\n(positive = faster at home)", fontweight="bold")
    add_tight_xaxis(axes[1], sp["home_road_delta"], n_ticks=10, decimals=2)
    axes[1].legend(fontsize=8)

    plt.suptitle("2025 MLB Pitcher 4-Seam Fastball Velocity", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig("fastball_velocity.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  → saved fastball_velocity.png")

    return overall, split


# ─────────────────────────────────────────────────────────────────────────────
# SHARED: compute GB%, FB%, LD%, K% for pitcher or batter side
# ─────────────────────────────────────────────────────────────────────────────

# PA-ending events (one row = one PA outcome)
PA_EVENTS = {
    "strikeout","strikeout_double_play",
    "walk","hit_by_pitch","intent_walk",
    "single","double","triple","home_run",
    "field_out","grounded_into_double_play","force_out",
    "double_play","triple_play","fielders_choice",
    "fielders_choice_out","sac_fly","sac_bunt",
    "sac_fly_double_play","catcher_interf",
}
K_EVENTS = {"strikeout","strikeout_double_play"}


def _k_rate(sub):
    pa = len(sub)
    return sub["events"].isin(K_EVENTS).sum() / pa * 100 if pa > 0 else np.nan


def _bb_rates(sub):
    n = len(sub)
    if n == 0:
        return pd.Series({"gb_pct": np.nan, "fb_pct": np.nan, "ld_pct": np.nan})
    return pd.Series({
        "gb_pct": (sub["bb_type"] == "ground_ball").sum() / n * 100,
        "fb_pct": (sub["bb_type"].isin(["fly_ball","popup"])).sum() / n * 100,
        "ld_pct": (sub["bb_type"] == "line_drive").sum() / n * 100,
    })


def compute_rates(data, group_col, home_col):
    """Return (overall_df, split_df) with K%, GB%, FB%, LD% for each team."""

    pa_df  = data[data["events"].isin(PA_EVENTS)].copy()
    bip_df = data[data["bb_type"].notna()].copy()

    # Overall
    k_ov  = pa_df.groupby(group_col).apply(_k_rate).reset_index().rename(columns={0:"k_pct"})
    bb_ov = bip_df.groupby(group_col).apply(_bb_rates).reset_index()
    overall = k_ov.merge(bb_ov, on=group_col)

    # Split
    def split_metric(df, fn, col_name_home, col_name_away):
        h = df[df[home_col]].groupby(group_col).apply(fn).reset_index().rename(columns={0: col_name_home})
        a = df[~df[home_col]].groupby(group_col).apply(fn).reset_index().rename(columns={0: col_name_away})
        return h.merge(a, on=group_col)

    k_split  = split_metric(pa_df,  _k_rate,   "k_home",  "k_away")
    gb_split = (
        bip_df[bip_df[home_col]].groupby(group_col).apply(_bb_rates).reset_index()
        .rename(columns={"gb_pct":"gb_home","fb_pct":"fb_home","ld_pct":"ld_home"})
        .merge(
            bip_df[~bip_df[home_col]].groupby(group_col).apply(_bb_rates).reset_index()
            .rename(columns={"gb_pct":"gb_away","fb_pct":"fb_away","ld_pct":"ld_away"}),
            on=group_col
        )
    )

    split = k_split.merge(gb_split, on=group_col)
    for m in ["k","gb","fb","ld"]:
        split[f"{m}_delta"] = split[f"{m}_home"] - split[f"{m}_away"]

    return overall, split


def print_rates_summary(side_label, group_col, overall, split):
    col_ov    = overall[overall[group_col] == COL_ABBR].iloc[0]
    col_split = split[split[group_col]     == COL_ABBR].iloc[0]
    print(f"\n  {side_label} rates for {COL_ABBR}:")
    for m, label in [("k","K%"),("gb","GB%"),("fb","FB%"),("ld","LD%")]:
        ov_v  = col_ov[f"{m}_pct"]
        h_v   = col_split[f"{m}_home"]
        a_v   = col_split[f"{m}_away"]
        d_v   = col_split[f"{m}_delta"]
        mlb_d = split[f"{m}_delta"].mean()
        print(f"    {label:5s}  overall={ov_v:5.1f}%  "
              f"home={h_v:5.1f}%  road={a_v:5.1f}%  "
              f"Δ={d_v:+.1f}pp  (MLB avg Δ={mlb_d:+.1f}pp, excess={d_v-mlb_d:+.1f}pp)")


def plot_rates(overall, split, group_col, title_prefix, filename_prefix):
    """4-row × 2-col figure: overall + home/road delta for K%, GB%, FB%, LD%."""

    metrics = [
        ("k_pct",  "k_delta",  "K%",  "K% Δ Home−Road"),
        ("gb_pct", "gb_delta", "GB%", "GB% Δ Home−Road"),
        ("fb_pct", "fb_delta", "FB%", "FB% Δ Home−Road"),
        ("ld_pct", "ld_delta", "LD%", "LD% Δ Home−Road"),
    ]

    fig, axes = plt.subplots(4, 2, figsize=(18, 26))

    for row_i, (ov_col, d_col, label, d_label) in enumerate(metrics):
        # Left: overall
        ax = axes[row_i, 0]
        df_s = overall.sort_values(ov_col)
        ax.barh(df_s[group_col], df_s[ov_col], color=team_colors(df_s[group_col]))
        mlb_avg = df_s[ov_col].mean()
        ax.axvline(mlb_avg, color="black", linestyle="--", linewidth=1,
                   label=f"MLB avg: {mlb_avg:.1f}%")
        ax.set_xlabel(f"{label} (%)", fontweight="bold")
        ax.set_title(f"Overall {label}", fontweight="bold")
        add_tight_xaxis(ax, df_s[ov_col], n_ticks=8, decimals=1)
        ax.legend(fontsize=8)

        # Right: delta
        ax = axes[row_i, 1]
        df_d = split.sort_values(d_col)
        ax.barh(df_d[group_col], df_d[d_col], color=team_colors(df_d[group_col]))
        mlb_d = df_d[d_col].mean()
        ax.axvline(0, color="black", linewidth=0.8)
        ax.axvline(mlb_d, color="red", linestyle="--", linewidth=1,
                   label=f"MLB avg Δ: {mlb_d:+.1f}pp")
        ax.set_xlabel(f"{d_label} (percentage points)", fontweight="bold")
        ax.set_title(f"{d_label}\n(positive = higher at home)", fontweight="bold")
        add_tight_xaxis(ax, df_d[d_col], n_ticks=8, decimals=1)
        ax.legend(fontsize=8)

    plt.suptitle(f"2025 MLB {title_prefix} – K%, GB%, FB%, LD%",
                 fontsize=15, fontweight="bold", y=1.005)
    plt.tight_layout()
    fname = f"{filename_prefix}_rates.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  → saved {fname}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. PITCHER RATES
# ─────────────────────────────────────────────────────────────────────────────

def analyze_pitcher_rates(data):
    print("\n" + "="*60)
    print("3. PITCHER RATES — K%, GB%, FB%, LD%  (home vs. road)")
    print("="*60)
    overall, split = compute_rates(data, "pitcher_team", "pitcher_is_home")
    print_rates_summary("PITCHER", "pitcher_team", overall, split)
    plot_rates(overall, split, "pitcher_team", "Pitcher Rates", "pitcher")
    return overall, split


# ─────────────────────────────────────────────────────────────────────────────
# 4. HITTER EXIT VELOCITY
# ─────────────────────────────────────────────────────────────────────────────

def analyze_exit_velocity(data):
    print("\n" + "="*60)
    print("4. HITTER EXIT VELOCITY (home vs. road)")
    print("="*60)

    bip = data[data["launch_speed"].notna()].copy()

    overall = (
        bip.groupby("batter_team")["launch_speed"]
        .mean().reset_index()
        .rename(columns={"launch_speed": "avg_ev"})
        .sort_values("avg_ev", ascending=False).reset_index(drop=True)
    )
    overall["rank"] = range(1, len(overall)+1)

    col_ov = overall[overall["batter_team"] == COL_ABBR].iloc[0]
    print(f"\n  COL overall avg EV: {col_ov['avg_ev']:.2f} mph  ({rank_label(col_ov['rank'])})")

    split = (
        bip.groupby(["batter_team","batter_is_home"])["launch_speed"]
        .mean().unstack()
        .rename(columns={True:"ev_home", False:"ev_away"})
        .reset_index()
    )
    split["ev_delta"] = split["ev_home"] - split["ev_away"]
    mlb_d = split["ev_delta"].mean()

    col_split = split[split["batter_team"] == COL_ABBR].iloc[0]
    print(f"\n  COL home EV: {col_split['ev_home']:.2f} mph  |  road EV: {col_split['ev_away']:.2f} mph")
    print(f"  COL EV Δ (H−R):            {col_split['ev_delta']:+.2f} mph")
    print(f"  MLB avg EV Δ (H−R):         {mlb_d:+.2f} mph")
    print(f"  COL excess Coors EV effect: {col_split['ev_delta'] - mlb_d:+.2f} mph")

    road_rank = (
        split[["batter_team","ev_away"]]
        .sort_values("ev_away", ascending=False).reset_index(drop=True)
    )
    road_rank["rank"] = range(1, len(road_rank)+1)
    col_road = road_rank[road_rank["batter_team"]==COL_ABBR].iloc[0]
    print(f"\n  COL road-adjusted EV rank: {rank_label(col_road['rank'])}  "
          f"({col_road['ev_away']:.2f} mph)")

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    ov = overall.sort_values("avg_ev")
    axes[0].barh(ov["batter_team"], ov["avg_ev"], color=team_colors(ov["batter_team"]))
    mlb_avg = overall["avg_ev"].mean()
    axes[0].axvline(mlb_avg, color="black", linestyle="--", linewidth=1,
                    label=f"MLB avg: {mlb_avg:.2f}")
    axes[0].set_xlabel("Avg Exit Velocity (mph)", fontweight="bold")
    axes[0].set_title("Overall Avg Exit Velocity by Team", fontweight="bold")
    add_tight_xaxis(axes[0], ov["avg_ev"], n_ticks=10, decimals=1)
    axes[0].legend(fontsize=8)

    sp = split.sort_values("ev_delta")
    axes[1].barh(sp["batter_team"], sp["ev_delta"], color=team_colors(sp["batter_team"]))
    axes[1].axvline(0, color="black", linewidth=0.8)
    axes[1].axvline(mlb_d, color="red", linestyle="--", linewidth=1,
                    label=f"MLB avg Δ: {mlb_d:+.2f}")
    axes[1].set_xlabel("EV Δ Home − Road (mph)", fontweight="bold")
    axes[1].set_title("Hitter EV Home vs. Road Delta\n(positive = hits harder at home)", fontweight="bold")
    add_tight_xaxis(axes[1], sp["ev_delta"], n_ticks=10, decimals=2)
    axes[1].legend(fontsize=8)

    plt.suptitle("2025 MLB Hitter Exit Velocity", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig("exit_velocity.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  → saved exit_velocity.png")

    return overall, split


# ─────────────────────────────────────────────────────────────────────────────
# 5. HITTER LAUNCH ANGLE
# ─────────────────────────────────────────────────────────────────────────────

def analyze_launch_angle(data):
    print("\n" + "="*60)
    print("5. HITTER LAUNCH ANGLE (home vs. road)")
    print("="*60)

    bip = data[data["launch_angle"].notna()].copy()

    overall = (
        bip.groupby("batter_team")["launch_angle"]
        .mean().reset_index()
        .rename(columns={"launch_angle": "avg_la"})
        .sort_values("avg_la", ascending=False).reset_index(drop=True)
    )
    overall["rank"] = range(1, len(overall)+1)

    col_ov = overall[overall["batter_team"] == COL_ABBR].iloc[0]
    print(f"\n  COL overall avg LA: {col_ov['avg_la']:.2f}°  ({rank_label(col_ov['rank'])})")
    print("  (optimal avg LA for hard contact is roughly 10–15°)")

    split = (
        bip.groupby(["batter_team","batter_is_home"])["launch_angle"]
        .mean().unstack()
        .rename(columns={True:"la_home", False:"la_away"})
        .reset_index()
    )
    split["la_delta"] = split["la_home"] - split["la_away"]
    mlb_d = split["la_delta"].mean()

    col_split = split[split["batter_team"] == COL_ABBR].iloc[0]
    print(f"\n  COL home LA: {col_split['la_home']:.2f}°  |  road LA: {col_split['la_away']:.2f}°")
    print(f"  COL LA Δ (H−R):             {col_split['la_delta']:+.2f}°")
    print(f"  MLB avg LA Δ (H−R):          {mlb_d:+.2f}°")
    print(f"  COL excess Coors LA effect:  {col_split['la_delta'] - mlb_d:+.2f}°")

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    ov = overall.sort_values("avg_la")
    axes[0].barh(ov["batter_team"], ov["avg_la"], color=team_colors(ov["batter_team"]))
    axes[0].axvline(overall["avg_la"].mean(), color="black", linestyle="--", linewidth=1)
    axes[0].axvline(12.5, color="green", linestyle=":", linewidth=1.2, label="Optimal zone ~10–15°")
    axes[0].set_xlabel("Avg Launch Angle (°)", fontweight="bold")
    axes[0].set_title("Overall Avg Launch Angle by Team", fontweight="bold")
    add_tight_xaxis(axes[0], ov["avg_la"], n_ticks=10, decimals=1)
    axes[0].legend(fontsize=8)

    sp = split.sort_values("la_delta")
    axes[1].barh(sp["batter_team"], sp["la_delta"], color=team_colors(sp["batter_team"]))
    axes[1].axvline(0, color="black", linewidth=0.8)
    axes[1].axvline(mlb_d, color="red", linestyle="--", linewidth=1,
                    label=f"MLB avg Δ: {mlb_d:+.2f}°")
    axes[1].set_xlabel("Launch Angle Δ Home − Road (°)", fontweight="bold")
    axes[1].set_title("Hitter Launch Angle Home vs. Road Delta", fontweight="bold")
    add_tight_xaxis(axes[1], sp["la_delta"], n_ticks=10, decimals=2)
    axes[1].legend(fontsize=8)

    plt.suptitle("2025 MLB Hitter Launch Angle", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig("launch_angle.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  → saved launch_angle.png")

    return overall, split


# ─────────────────────────────────────────────────────────────────────────────
# 6. HITTER RATES — K%, GB%, FB%, LD%
# ─────────────────────────────────────────────────────────────────────────────

def analyze_hitter_rates(data):
    print("\n" + "="*60)
    print("6. HITTER RATES — K%, GB%, FB%, LD%  (home vs. road)")
    print("="*60)
    overall, split = compute_rates(data, "batter_team", "batter_is_home")
    print_rates_summary("HITTER", "batter_team", overall, split)
    plot_rates(overall, split, "batter_team", "Hitter Rates", "hitter")
    return overall, split


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(results):
    print("\n" + "="*60)
    print("SUMMARY — 2025 COLORADO ROCKIES vs. ALL 30 MLB TEAMS")
    print("="*60)

    rows = []

    if results.get("sprint") is not None:
        df = results["sprint"]
        r  = df[df["team"]==COL_ABBR].iloc[0]
        rows.append(("Sprint Speed", f"{r['avg_sprint_speed']:.2f} ft/sec", rank_label(r["rank"])))

    if results.get("fb_overall") is not None:
        df = results["fb_overall"]
        r  = df[df["pitcher_team"]==COL_ABBR].iloc[0]
        rows.append(("4-Seam Velo (overall)", f"{r['avg_velo']:.2f} mph", rank_label(r["rank"])))

    if results.get("ev_overall") is not None:
        df = results["ev_overall"]
        r  = df[df["batter_team"]==COL_ABBR].iloc[0]
        rows.append(("Exit Velo (overall)", f"{r['avg_ev']:.2f} mph", rank_label(r["rank"])))

    if results.get("la_overall") is not None:
        df = results["la_overall"]
        r  = df[df["batter_team"]==COL_ABBR].iloc[0]
        rows.append(("Launch Angle (overall)", f"{r['avg_la']:.2f}°", rank_label(r["rank"])))

    for side_label, key, gc in [("Pitcher","p_rates","pitcher_team"),
                                  ("Hitter", "h_rates","batter_team")]:
        if results.get(key) is not None:
            ov_df, _ = results[key]
            col_r = ov_df[ov_df[gc]==COL_ABBR].iloc[0]
            for m, lbl in [("k","K%"),("gb","GB%"),("fb","FB%"),("ld","LD%")]:
                val = col_r[f"{m}_pct"]
                ranked = ov_df.sort_values(f"{m}_pct", ascending=False).reset_index(drop=True)
                ranked["_r"] = range(1, len(ranked)+1)
                rk = ranked[ranked[gc]==COL_ABBR].iloc[0]["_r"]
                rows.append((f"{side_label} {lbl}", f"{val:.1f}%", rank_label(int(rk))))

    print(f"\n  {'Metric':<30} {'COL Value':<18} Rank")
    print("  " + "-"*58)
    for m, v, r in rows:
        print(f"  {m:<30} {v:<18} {r}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    results = {}

    results["sprint"] = analyze_sprint_speed()

    sc = load_statcast_data()

    fb_overall, fb_split = analyze_fastball_velocity(sc)
    results["fb_overall"] = fb_overall

    p_ov, p_split = analyze_pitcher_rates(sc)
    results["p_rates"] = (p_ov, p_split)

    ev_overall, ev_split = analyze_exit_velocity(sc)
    results["ev_overall"] = ev_overall

    la_overall, la_split = analyze_launch_angle(sc)
    results["la_overall"] = la_overall

    h_ov, h_split = analyze_hitter_rates(sc)
    results["h_rates"] = (h_ov, h_split)

    print_summary(results)

    print("\nAll plots saved to working directory.")
    print("Run complete! ✓")


if __name__ == "__main__":
    main()
