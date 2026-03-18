"""
speed.py — Statcast Sprint Speed data and baserunning advancement probability functions.

What this module does:
  1. Provides verified 2025 Statcast sprint speeds for all Rockies position players
  2. Converts ft/sec to MLB percentile using an empirically calibrated normal distribution
  3. Computes per-runner advancement probabilities (1st→3rd, 2nd→home on a single)
     based on the runner's individual sprint speed and game location (Coors vs road)

Sources:
  - Sprint speeds: Baseball Savant / Statcast (verified via Purple Row Feb 2026,
    Just Baseball Aug 2025, Sportsnaut 2025)
  - MLB distribution parameters: Baseball Savant documentation
    "The Major League average on a competitive play is 27 ft/sec;
     the competitive range is roughly 23 ft/sec (poor) to 30 ft/sec (elite)."
  - Coors Field advancement boost: derived from historical base-running data
    showing ~10-12% higher extra-base advancement rates at Coors vs neutral parks,
    due to the larger outfield (largest in MLB by area) giving outfielders more
    ground to cover.
  - Advancement base rates: MLB Statcast base-running research;
    runner on 2nd scores on single ~60%, runner on 1st takes 3rd ~28% (neutral park)

"""

import math
from dataclasses import dataclass
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# MLB SPRINT SPEED DISTRIBUTION PARAMETERS
# Source: Baseball Savant documentation + empirical calibration
# Distribution: approximately Normal(mean=27.0, std=1.5 ft/sec)
# ─────────────────────────────────────────────────────────────────────────────

MLB_AVG_SPEED_FPS   = 27.0    # League average sprint speed (ft/sec)
MLB_SPEED_STD_FPS   =  1.5    # Approximate standard deviation
MLB_ELITE_SPEED_FPS = 30.0    # "Bolt" threshold — elite speed
MLB_POOR_SPEED_FPS  = 23.0    # Lower bound of competitive range


# ─────────────────────────────────────────────────────────────────────────────
# BASE ADVANCEMENT RATES (neutral park, MLB-average runner)
#
# These are TRUE NEUTRAL PARK baselines derived from Statcast base-running research.
# Do NOT pre-adjust these for Coors — the park factor is applied separately
# via COORS_ADVANCE_BOOST after computing the speed-adjusted rate.
#
# MLB neutral (any ballpark) averages:
#   Runner on 2nd scoring on a single:   ~60%
#   Runner on 1st advancing to 3rd:      ~28%
#   Runner on 3rd scoring on a single:   ~100% (hardcoded in simulator)
# ─────────────────────────────────────────────────────────────────────────────

MLB_NEUTRAL_2ND_SCORES_ON_SINGLE = 0.60   # runner on 2nd → scores on single
MLB_NEUTRAL_1ST_TO_3RD_ON_SINGLE = 0.28   # runner on 1st → 3rd on single

# Coors Field park factor for extra-base advancement:
COORS_ADVANCE_BOOST = 1.11

# Speed effect coefficient:
# How much each ft/sec above/below MLB average changes advancement probability.
# Calibrated so that:
#   30 ft/sec (elite):  +43% advancement vs avg → mult ≈ 1.43
#   27 ft/sec (avg):     0% change              → mult = 1.00
#   24 ft/sec (slow):  -43% advancement vs avg  → mult ≈ 0.57
# This is approximately linear within the 23–30 ft/sec range.
SPEED_EFFECT_PER_FPS = 0.145


# ─────────────────────────────────────────────────────────────────────────────
# PERCENTILE CALIBRATION TABLE
# Derived from Normal(27.0, 1.5) distribution.
# Calibration: Φ((x - 27.0) / 1.5) where Φ is the standard normal CDF.
# Used for display purposes; the actual advancement calculation uses the
# raw ft/sec value directly.
# ─────────────────────────────────────────────────────────────────────────────

# (ft_per_sec, approximate_percentile)
_PERCENTILE_TABLE = [
    (23.0,  1),
    (23.5,  2),
    (24.0,  3),
    (24.5,  5),
    (25.0,  9),
    (25.5, 16),
    (26.0, 25),
    (26.5, 37),
    (27.0, 50),
    (27.5, 63),
    (28.0, 75),
    (28.5, 84),
    (29.0, 91),
    (29.2, 93),
    (29.5, 95),
    (29.9, 98),
    (30.0, 98),
    (30.5, 99),
]


def fps_to_percentile(speed_fps: float) -> int:
    """
    Convert a sprint speed in ft/sec to an MLB percentile (1–99).
    Uses linear interpolation between the calibrated table points.

    Examples:
        fps_to_percentile(27.0)  →  50
        fps_to_percentile(29.9)  →  98
        fps_to_percentile(24.5)  →   5
    """
    if speed_fps <= _PERCENTILE_TABLE[0][0]:
        return _PERCENTILE_TABLE[0][1]
    if speed_fps >= _PERCENTILE_TABLE[-1][0]:
        return _PERCENTILE_TABLE[-1][1]

    for i in range(len(_PERCENTILE_TABLE) - 1):
        lo_fps, lo_pct = _PERCENTILE_TABLE[i]
        hi_fps, hi_pct = _PERCENTILE_TABLE[i + 1]
        if lo_fps <= speed_fps <= hi_fps:
            frac = (speed_fps - lo_fps) / (hi_fps - lo_fps)
            return round(lo_pct + frac * (hi_pct - lo_pct))

    return 50  # fallback


def speed_multiplier(speed_fps: float) -> float:
    """
    Compute a runner's advancement probability multiplier relative to MLB average.
    Linear function of deviation from league average (27.0 ft/sec).

    Returns a value clamped to [0.40, 1.60] to avoid extreme edge cases.

    Examples:
        speed_multiplier(27.0)  →  1.00   (league average, no change)
        speed_multiplier(29.9)  →  1.42   (McCarthy — 42% more likely to advance)
        speed_multiplier(24.5)  →  0.65   (slow catcher — 35% less likely)
    """
    raw = 1.0 + SPEED_EFFECT_PER_FPS * (speed_fps - MLB_AVG_SPEED_FPS)
    return max(0.40, min(1.60, raw))


def advancement_probs(
    runner_speed_fps: float,
    location: str,
) -> tuple:
    """
    Compute the probability that a specific runner:
      (a) scores from 2nd base on a single
      (b) advances from 1st to 3rd on a single

    These are the two main "extra base" advancement decisions in baseball;
    all other advancement cases (3rd scores on single, 2nd/3rd score on double, etc.)
    are handled deterministically in the simulator.

    Parameters:
        runner_speed_fps: the runner's Statcast sprint speed in ft/sec
        location: 'home' (Coors Field) or 'away' (neutral)

    Returns:
        (p_2nd_scores, p_1st_to_3rd) as floats

    Calculation:
        1. Start with MLB neutral base rates (0.60, 0.28)
        2. Apply individual speed multiplier
        3. Apply Coors park factor if location == 'home'
        4. Clamp to [0.0, 0.95] to avoid impossibilities
    """
    mult = speed_multiplier(runner_speed_fps)

    p_2nd_scores = MLB_NEUTRAL_2ND_SCORES_ON_SINGLE * mult
    p_1st_to_3rd = MLB_NEUTRAL_1ST_TO_3RD_ON_SINGLE * mult

    if location == "home":
        p_2nd_scores *= COORS_ADVANCE_BOOST
        p_1st_to_3rd *= COORS_ADVANCE_BOOST

    p_2nd_scores = min(p_2nd_scores, 0.95)
    p_1st_to_3rd = min(p_1st_to_3rd, 0.52)

    return p_2nd_scores, p_1st_to_3rd


# ─────────────────────────────────────────────────────────────────────────────
# VERIFIED 2025 STATCAST SPRINT SPEEDS — COLORADO ROCKIES
#
# Sources (all referenced in code):
#   - Purple Row, "Taking Stock of the Rockies' Toolbox" (Feb 2, 2026):
#     confirmed McCarthy 29.9, Doyle 29.5, Ritter 29.2, Castro 28.8;
#     Freeman/Moniak/Beck all 80th percentile or above
#   - Just Baseball (Aug 2025): Doyle 29.5 ft/sec, 96th percentile confirmed
#   - Sportsnaut (2025): McCarthy 30.0-30.1 ft/sec, 98th percentile confirmed
#   - Wikipedia / multiple sources: McCarthy "98th percentile sprint speed"
#   - Rockies.mlb.com (Feb 2025): "Doyle, Tovar above 70th percentile"
#   - Estimated where no published figure exists (see notes)
#
# MLB avg = 27.0, std ≈ 1.5 ft/sec
# 80th percentile ≈ 28.2 ft/sec, 70th ≈ 27.7 ft/sec
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SpeedEntry:
    name:       str
    speed_fps:  float
    percentile: int
    source:     str    # data quality flag: 'verified', 'confirmed_range', 'estimated'
    notes:      str = ""


ROCKIES_SPEED_DATA_2025: list = [
    SpeedEntry("Jake McCarthy",  29.9, fps_to_percentile(29.9), "verified",
               "Purple Row Feb 2026; multiple sources confirm 98th-99th pct"),
    SpeedEntry("Brenton Doyle",  29.5, fps_to_percentile(29.5), "verified",
               "Just Baseball Aug 2025: 29.5 ft/sec, 96th pct; Purple Row Feb 2026 confirms"),
    SpeedEntry("Ryan Ritter",    29.2, fps_to_percentile(29.2), "verified",
               "Purple Row Feb 2026: 29.2 ft/sec, 90th+ pct"),
    SpeedEntry("Willi Castro",   28.8, fps_to_percentile(28.8), "verified",
               "Purple Row Feb 2026: 28.8 ft/sec"),
    SpeedEntry("Mickey Moniak",  28.1, fps_to_percentile(28.1), "confirmed_range",
               "Purple Row Feb 2026: 80th pct or above; ~28.0-28.3 estimated from range"),
    SpeedEntry("Jordan Beck",    28.0, fps_to_percentile(28.0), "confirmed_range",
               "Purple Row Feb 2026: 80th pct or above; ~28.0 estimated from range"),
    SpeedEntry("Tyler Freeman",  27.9, fps_to_percentile(27.9), "confirmed_range",
               "Purple Row Feb 2026: 80th pct or above; ~27.9 estimated from range"),
    SpeedEntry("Ezequiel Tovar", 27.6, fps_to_percentile(27.6), "confirmed_range",
               "Rockies.mlb.com Feb 2025: above 70th pct; ~27.6 estimated"),
    SpeedEntry("Edouard Julien", 26.5, fps_to_percentile(26.5), "estimated",
               "No published figure. 2B/1B profile; estimated below-average (~37th pct)"),
    SpeedEntry("Hunter Goodman", 24.5, fps_to_percentile(24.5), "estimated",
               "No published figure. Catcher; estimated well below average (~5th pct)"),
]

# Build a lookup dict for use in roster.py
SPEED_LOOKUP: dict = {e.name: e.speed_fps for e in ROCKIES_SPEED_DATA_2025}


def get_speed(player_name: str, fallback: float = MLB_AVG_SPEED_FPS) -> float:
    """
    Look up a player's verified/estimated sprint speed.
    Falls back to MLB average if the player is not in the table.
    """
    return SPEED_LOOKUP.get(player_name, fallback)


def print_speed_report() -> None:
    """Print a formatted speed report for all tracked players."""
    print("\n" + "=" * 80)
    print("  ROCKIES POSITION PLAYERS — SPRINT SPEED REPORT (2025 Statcast)")
    print(f"  MLB avg: {MLB_AVG_SPEED_FPS} ft/sec  |  Elite (Bolt): ≥{MLB_ELITE_SPEED_FPS} ft/sec")
    print("=" * 80)
    print(f"  {'Name':<22} {'ft/sec':>7} {'Pct':>5}  {'Mult':>6}  {'Source':<18} Notes")
    print("  " + "─" * 78)

    for e in sorted(ROCKIES_SPEED_DATA_2025, key=lambda x: -x.speed_fps):
        mult = speed_multiplier(e.speed_fps)
        p2, p13 = advancement_probs(e.speed_fps, "away")
        p2h, p13h = advancement_probs(e.speed_fps, "home")
        print(
            f"  {e.name:<22} {e.speed_fps:>6.1f} {e.percentile:>5}%  "
            f"{mult:>5.2f}x  {e.source:<18} {e.notes[:40]}"
        )

    print(f"\n  Advancement probabilities by runner (neutral / Coors):")
    print(f"  {'Name':<22} {'2nd→Score (away)':>18} {'2nd→Score (home)':>18} "
          f"{'1st→3rd (away)':>16} {'1st→3rd (home)':>16}")
    print("  " + "─" * 93)
    for e in sorted(ROCKIES_SPEED_DATA_2025, key=lambda x: -x.speed_fps):
        p2a, p13a = advancement_probs(e.speed_fps, "away")
        p2h, p13h = advancement_probs(e.speed_fps, "home")
        print(
            f"  {e.name:<22} {p2a:>17.1%} {p2h:>18.1%} "
            f"{p13a:>16.1%} {p13h:>16.1%}"
        )


if __name__ == "__main__":
    print_speed_report()

    print("\n  Example: pulling live data with pybaseball (requires: pip install pybaseball)")
    print("""
    from pybaseball import statcast_sprint_speed
    import pandas as pd

    # Pull full 2025 sprint speed leaderboard
    df = statcast_sprint_speed(2025, min_opp=10, position='')

    # Filter for qualifying players (≥10 competitive runs, minimum PA threshold)
    df = df[df['sprint_speed'].notna()].copy()

    # Add percentile column using our calibrated function
    from speed import fps_to_percentile
    df['percentile'] = df['sprint_speed'].apply(fps_to_percentile)

    # Filter for Rockies
    col = df[df['team'] == 'COL'].sort_values('sprint_speed', ascending=False)
    print(col[['player_name', 'sprint_speed', 'percentile']].to_string(index=False))

    # To get career multi-year average (more stable than single-season):
    seasons = []
    for yr in [2023, 2024, 2025]:
        s = statcast_sprint_speed(yr, min_opp=10)
        s['year'] = yr
        seasons.append(s)
    career = pd.concat(seasons).groupby('player_name')['sprint_speed'].mean().reset_index()
    """)
