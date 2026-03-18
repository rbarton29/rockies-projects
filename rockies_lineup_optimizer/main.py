"""
main.py — Entry point for the Colorado Rockies Lineup Optimizer.

Runs all four splits:
  Home vs RHP  |  Home vs LHP  |  Away vs RHP  |  Away vs LHP

Usage:
    python main.py                              # 2026 roster, all 4 splits
    python main.py --year 2025                  # 2025 roster
    python main.py --top 15 --swaps 500         # 15 lineups, 500 swap iterations
    python main.py --games 2000 --eval 400      # higher precision
    python main.py --outdir ./results           # custom output directory

Output files per split (in --outdir):
    rockies_top_lineups_<split>.csv
    rockies_slot_freq_<split>.csv
"""

import argparse
from pathlib import Path

from roster import get_roster_2025, get_roster_2026
from stats import build_profiles
from simulator import FixedPitcherSimulator
from optimizer import LineupOptimizer
from reporting import (
    print_roster_summary,
    print_top_lineups,
    print_run_distribution,
    save_top_lineups_csv,
    save_slot_frequency_csv,
)


# ─────────────────────────────────────────────────────────────────────────────
# SPLIT DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────

SPLITS = [
    # (location, pitcher_hand, display_label)
    ("home", "R", "Home vs RHP"),
    ("home", "L", "Home vs LHP"),
    ("away", "R", "Away vs RHP"),
    ("away", "L", "Away vs LHP"),
]


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Rockies Lineup Optimizer — 4 splits")
    ap.add_argument("--year",   type=int, default=2026, choices=[2025, 2026],
                    help="Roster year (default: 2026)")
    ap.add_argument("--top",    type=int, default=10,
                    help="Top N lineups to surface per split (default: 10)")
    ap.add_argument("--swaps",  type=int, default=400,
                    help="Hill-climbing swap iterations per split (default: 400)")
    ap.add_argument("--eval",   type=int, default=300,
                    help="Games per quick eval during search (default: 300)")
    ap.add_argument("--games",  type=int, default=1500,
                    help="Games for final high-precision re-ranking (default: 1500)")
    ap.add_argument("--outdir", type=str, default=".",
                    help="Output directory for CSV files (default: current dir)")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 90)
    print(f"  COLORADO ROCKIES LINEUP OPTIMIZER — {args.year} PROJECTIONS")
    print("  4 Splits: Home vs RHP | Home vs LHP | Away vs RHP | Away vs LHP")
    print("=" * 90)

    # Build player profiles
    roster = get_roster_2026() if args.year == 2026 else get_roster_2025()
    profiles = build_profiles(roster)

    # Print full stat summary
    print_roster_summary(profiles)

    # ── Run optimizer for each split ─────────────────────────────────────────
    all_results: dict = {}   # split_label → (ranked, sim_result_of_#1)

    for location, pitcher_hand, label in SPLITS:
        print(f"\n{'=' * 90}")
        print(f"  SPLIT: {label}")
        print(f"  swaps={args.swaps} | eval_games={args.eval} | final_games={args.games}")
        print(f"{'=' * 90}")

        sim       = FixedPitcherSimulator(location, pitcher_hand)
        optimizer = LineupOptimizer(
            simulator        = sim,
            n_swaps          = args.swaps,
            n_eval_games     = args.eval,
            n_final_games    = args.games,
            top_n            = args.top,
            accept_threshold = 0.020,
            seed             = 42,
        )

        ranked = optimizer.optimize(profiles)

        print_top_lineups(ranked, label)
        print_run_distribution(ranked[0][1], label=f"#{1} lineup — {label}")

        # Save CSVs
        save_top_lineups_csv(ranked, label, location, pitcher_hand, outdir)
        save_slot_frequency_csv(ranked, label, outdir)

        all_results[label] = ranked

    # ── Season projection summary ─────────────────────────────────────────────
    print(f"\n{'=' * 90}")
    print(f"  SEASON PROJECTION SUMMARY ({args.year})")
    print(f"{'=' * 90}")
    print(f"  {'Split':<18} {'#1 Lineup R/G':>14} {'Proj Runs/81G':>15}")
    print("  " + "─" * 50)

    home_runs = 0.0
    away_runs = 0.0

    for location, pitcher_hand, label in SPLITS:
        ranked = all_results.get(label, [])
        if not ranked:
            continue
        r_g    = ranked[0][1]["mean"]
        runs81 = r_g * 81
        # Rough weighting: ~70% of games vs RHP starters, 30% vs LHP
        weight = 0.70 if pitcher_hand == "R" else 0.30
        if location == "home":
            home_runs += r_g * weight
        else:
            away_runs += r_g * weight
        print(f"  {label:<18} {r_g:>14.3f} {runs81:>15.0f}")

    total_runs = (home_runs + away_runs) * 81
    print(f"\n  Weighted home R/G  (70% RHP / 30% LHP): {home_runs:.3f}")
    print(f"  Weighted away R/G  (70% RHP / 30% LHP): {away_runs:.3f}")
    print(f"  Coors advantage: +{home_runs - away_runs:.3f} R/G per home game")
    print(f"  Projected season runs (162 G):           {total_runs:.0f}")
    print(f"\n  CSV files written to: {outdir.resolve()}")
    print("\n  Done.\n")


if __name__ == "__main__":
    main()
