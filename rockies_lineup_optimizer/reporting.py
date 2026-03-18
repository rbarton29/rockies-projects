"""
reporting.py — Console output and CSV export.

Print functions:
  print_roster_summary()   — full stat table per split, hitter type classifier
  print_top_lineups()      — ranked lineups with diversity scores
  print_run_distribution() — ASCII histogram

CSV output:
  save_top_lineups_csv()    — one row per lineup, slot-by-slot breakdown
  save_slot_frequency_csv() — player × slot heatmap across top-N lineups
"""

import csv
from collections import defaultdict
from pathlib import Path

from stats import PlayerProfile, ProbSet
from optimizer import diversity_score, slot_frequency


# ─────────────────────────────────────────────────────────────────────────────
# ROSTER SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def _hitter_type(probs: ProbSet, gb_fb: float, sb: int) -> str:
    """
    Classify hitter profile based on OBP, ISO, Power Factor, GB/FB ratio, and SB.

    GB/FB ratio:
      > 1.5  → ground-ball hitter (contact, fewer HRs)
      0.8–1.5 → balanced
      < 0.8  → fly-ball hitter (more power, more HR)

    ISO:
      >= 0.200 → plus power
      0.130–0.199 → average power
      < 0.130 → below-average power

    PWR (XBH/H):
      >= 0.40 → most hits go for extra bases (true slugger)
      < 0.25  → mostly singles (contact/table-setter)
    """
    iso   = probs.ISO
    pwr   = probs.PWR
    obp   = probs.OBP

    if obp >= 0.370 and iso < 0.130:
        return "Leadoff / OBP"
    elif iso >= 0.200 and pwr >= 0.40:
        return "Power Slugger"
    elif iso >= 0.150 and obp >= 0.340:
        return "Middle-order"
    elif gb_fb >= 1.5:
        return "Ground-ball"
    elif sb >= 10 and obp >= 0.330:
        return "Speed / Table-set"
    elif pwr >= 0.35:
        return "Gap / Extra-base"
    else:
        return "Contact / Utility"


def print_roster_summary(profiles: list) -> None:
    """
    Print stat tables for all four splits with ISO, PWR, GB/FB columns,
    plus a hitter-type classification summary.
    """
    SPLITS = [
        ("home", "R", "HOME vs RHP (Coors Field)"),
        ("home", "L", "HOME vs LHP (Coors Field)"),
        ("away", "R", "AWAY vs RHP"),
        ("away", "L", "AWAY vs LHP"),
    ]

    print("\n" + "=" * 115)
    print("  2026 COLORADO ROCKIES — PLAYER STAT PROFILES")
    print("  ISO = SLG - AVG  |  PWR = XBH/H  |  GB/FB = groundball-to-flyball ratio")
    print("=" * 115)

    for location, hand, label in SPLITS:
        print(f"\n  ── {label} ──")
        hdr = (
            f"  {'Name':<22} {'POS':<6} {'G':<4} "
            f"{'AVG':<6} {'OBP':<6} {'SLG':<6} {'OPS':<6} "
            f"{'ISO':<6} {'PWR':<6} {'GB/FB':<7} "
            f"{'Spd':>5} {'Pct':>4} "
            f"{'HR':<4} {'SB':<4} {'Splits?'}"
        )
        print(hdr)
        print("  " + "─" * 108)

        from speed import fps_to_percentile
        key = f"{location}_vs_{hand}"
        for p in sorted(profiles, key=lambda x: getattr(x, key).OPS, reverse=True):
            pr   = getattr(p, key)
            gbfb = f"{p.gb_fb_ratio():.2f}"
            v    = "✓" if p.verified_splits else "~"
            spd  = p.player.speed_fps
            pct  = fps_to_percentile(spd)
            print(
                f"  {p.name:<22} {p.pos:<6} {p.games:<4} "
                f"{pr.AVG:.3f}  {pr.OBP:.3f}  {pr.SLG:.3f}  {pr.OPS:.3f}  "
                f"{pr.ISO:.3f}  {pr.PWR:.3f}  {gbfb:<7} "
                f"{spd:>5.1f} {pct:>3}%  "
                f"{pr.HR_count:<4} {pr.SB:<4} {v}"
            )

    # ── Hitter type summary ──────────────────────────────────────────────────
    print(f"\n  ── Hitter Type Classification (based on HOME vs RHP) ──")
    print(
        f"  {'Name':<22} {'Type':<20} {'ISO':<7} {'PWR':<7} "
        f"{'GB/FB':<7} {'OPS':<7} {'OPS gap H-A'}"
    )
    print("  " + "─" * 85)
    for p in sorted(profiles, key=lambda x: x.home_vs_R.OPS, reverse=True):
        hp   = p.home_vs_R
        ap   = p.away_vs_R
        gbfb = p.gb_fb_ratio()
        htype = _hitter_type(hp, gbfb, hp.SB)
        gap   = hp.OPS - ap.OPS
        print(
            f"  {p.name:<22} {htype:<20} {hp.ISO:.3f}  {hp.PWR:.3f}  "
            f"{gbfb:.2f}   {hp.OPS:.3f}  {gap:+.3f}"
        )

    # ── Notes / warnings ──────────────────────────────────────────────────────
    print(f"\n  ── Projection Notes ──")
    for p in profiles:
        if p.notes:
            flag = "⚠ " if "⚠" in p.notes else "  "
            print(f"  {flag}{p.name}: {p.notes}")


# ─────────────────────────────────────────────────────────────────────────────
# TOP LINEUPS PRINT
# ─────────────────────────────────────────────────────────────────────────────

def print_top_lineups(ranked: list, split_label: str) -> None:
    """
    Print a ranked table of top lineups with diversity and run stats.
    ranked: list of (lineup, sim_result_dict)
    """
    if not ranked:
        return

    ref_lineup = ranked[0][0]
    n = len(ranked)

    print(f"\n{'=' * 105}")
    print(f"  TOP {n} LINEUPS — {split_label}")
    print(f"{'=' * 105}")
    print(
        f"  {'#':<4} {'R/G':>6} {'Std':>5} {'P10–P90':<12} "
        f"{'Slots ≠ #1':<12} Order (last names)"
    )
    print("  " + "─" * 100)

    for rank, (lineup, sim) in enumerate(ranked, 1):
        diff   = diversity_score(lineup, ref_lineup) if rank > 1 else 0
        order  = " → ".join(p.name.split()[-1] for p in lineup)
        p_range = f"{sim['p10']:.1f}–{sim['p90']:.1f}"
        diff_s  = f"{diff} slots" if rank > 1 else "(reference)"
        print(
            f"  {rank:<4} {sim['mean']:>6.3f} {sim['std']:>5.2f} "
            f"{p_range:<12} {diff_s:<12} {order}"
        )

    # Slot frequency summary
    freq = slot_frequency(ranked)
    print(f"\n  Slot frequency across top {n} lineups:")
    print(f"  {'Name':<22} " + "  ".join(f"#{s}" for s in range(1, 10)))
    print("  " + "─" * 80)
    for p_name, slots in sorted(freq.items(), key=lambda x: -max(x[1].values())):
        row = "  ".join(f"{slots.get(s, 0):>2}" for s in range(1, 10))
        most = max(slots, key=slots.get)
        print(f"  {p_name:<22} {row}   (most often: #{most})")


# ─────────────────────────────────────────────────────────────────────────────
# RUN DISTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────

def print_run_distribution(sim_result: dict, label: str = "") -> None:
    hist  = sim_result["hist"]
    total = sum(hist)
    if total == 0:
        return
    mode_count = max(hist)
    print(f"\n  Run distribution — {label} (n={sim_result['n']})")
    for runs, count in enumerate(hist):
        if count == 0:
            continue
        pct    = count / total * 100
        bar    = "█" * int(pct * 1.6)
        marker = " ← mode" if count == mode_count else ""
        print(f"  {runs:>2} runs: {bar:<30} {pct:4.1f}%{marker}")
    print(
        f"  Mean {sim_result['mean']:.3f}  |  "
        f"Std {sim_result['std']:.3f}  |  "
        f"P10–P90 {sim_result['p10']:.1f}–{sim_result['p90']:.1f}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# CSV EXPORT
# ─────────────────────────────────────────────────────────────────────────────

def save_top_lineups_csv(
    ranked:      list,
    split_label: str,
    location:    str,
    pitcher_hand: str,
    outdir:      Path,
) -> Path:
    """
    One row per lineup. Columns:
      rank, split, avg_R_G, std, p10, p90, slots_diff_vs_1, sim_games,
      slot_1 … slot_9 (name + pos + OPS + ISO + OBP + PWR + GBFB per slot)
    """
    ref_lineup = ranked[0][0] if ranked else []
    probs_key  = f"{location}_vs_{pitcher_hand}"

    rows = []
    for rank, (lineup, sim) in enumerate(ranked, 1):
        diff = diversity_score(lineup, ref_lineup) if rank > 1 else 0
        row  = {
            "rank":             rank,
            "split":            split_label,
            "avg_runs_per_game": round(sim["mean"], 4),
            "std":              round(sim["std"],  4),
            "p10":              round(sim["p10"],  2),
            "p90":              round(sim["p90"],  2),
            "slots_diff_vs_1":  diff,
            "sim_games":        sim["n"],
        }
        for slot, player in enumerate(lineup, 1):
            pr = getattr(player, probs_key)
            gbfb = round(player.gb_fb_ratio(), 2)
            row[f"slot_{slot}_name"] = player.name
            row[f"slot_{slot}_pos"]  = player.pos
            row[f"slot_{slot}_ops"]  = round(pr.OPS, 3)
            row[f"slot_{slot}_iso"]  = round(pr.ISO, 3)
            row[f"slot_{slot}_obp"]  = round(pr.OBP, 3)
            row[f"slot_{slot}_pwr"]  = round(pr.PWR, 3)
            row[f"slot_{slot}_gbfb"] = gbfb
        rows.append(row)

    slug = split_label.lower().replace(" ", "_").replace("/", "_")
    path = outdir / f"rockies_top_lineups_{slug}.csv"
    if rows:
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    print(f"  Saved: {path}")
    return path


def save_slot_frequency_csv(
    ranked:      list,
    split_label: str,
    outdir:      Path,
) -> Path:
    """
    Player × slot frequency heatmap across top-N lineups.
    High counts in one slot → optimizer is confident about that player's position.
    Spread across slots → genuine ambiguity (look at run totals to understand why).
    """
    freq = slot_frequency(ranked)
    all_names = list(freq.keys())

    rows = []
    for name in sorted(all_names):
        slots = freq[name]
        row   = {"player": name}
        for s in range(1, 10):
            row[f"slot_{s}"] = slots.get(s, 0)
        row["most_common_slot"] = max(range(1, 10), key=lambda s: slots.get(s, 0))
        rows.append(row)

    slug = split_label.lower().replace(" ", "_").replace("/", "_")
    path = outdir / f"rockies_slot_freq_{slug}.csv"
    if rows:
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    print(f"  Saved: {path}")
    return path
