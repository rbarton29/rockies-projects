"""
optimizer.py — Lineup optimizer with positional constraints and archetype seeding.

  - Positional constraint validation: any 9-player lineup must cover all 9
    defensive slots using each player's eligible_pos list.
  - Archetype-seeded starting order: instead of pure OPS-sort (which put slow
    catchers like Goodman in the leadoff spot), candidates are scored separately
    for each batting slot archetype before the hill-climb begins.
  - Roster swap logic: 30% of iterations swap a bench player in for an active
    player, exploring which 9 of the 12 should start.
  - Locked players: players flagged locked=True are never moved to the bench
    during roster swaps (e.g. Goodman, Karros, Tovar).
    
"""

import random
from collections import defaultdict

import numpy as np
from tqdm import tqdm

from simulator import GameSimulator


# ─────────────────────────────────────────────────────────────────────────────
# POSITION COVERAGE VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

REQUIRED_POSITIONS = {"C", "1B", "2B", "3B", "SS", "LF", "CF", "RF", "DH"}


def is_valid_lineup(lineup: list) -> bool:
    """
    Check whether a 9-player lineup covers all 9 required positions.
    Uses each player's eligible_pos list. A player can only be assigned
    to one position (no double-counting).

    Uses most-constrained-first ordering: positions with fewer eligible
    players are assigned first. This avoids the failure mode where a greedy
    left-to-right pass using a set's non-deterministic order assigns versatile
    players (e.g. Castro) to easy positions early, then can't fill hard ones
    (e.g. C, SS) with the remaining players.

    Example of the bug this fixes:
      If DH is processed before SS, Castro (eligible for both) might be
      assigned to DH, leaving nobody for SS.
      With most-constrained-first: C is processed first (1 eligible player),
      then SS (2-3 eligible), etc., so versatile players are always available
      for the positions that need them most.
    """
    if len(lineup) != 9:
        return False

    required = list(REQUIRED_POSITIONS)

    # Sort required positions: fewest eligible players first (most constrained)
    required.sort(key=lambda pos: sum(1 for p in lineup if pos in p.player.eligible_pos))

    assigned = set()

    for pos in required:
        filled = False
        for i, player in enumerate(lineup):
            if i not in assigned and pos in player.player.eligible_pos:
                assigned.add(i)
                filled = True
                break
        if not filled:
            return False

    return True


# ─────────────────────────────────────────────────────────────────────────────
# BATTING ORDER ARCHETYPE SCORING
#
# Assigns each player an archetype score for each batting slot.
# The optimizer starts from this seeded order rather than pure OPS-sort,
# which produced nonsensical results (e.g. slow catchers leading off).
#
# Slot archetypes:
#   1: Leadoff   — high OBP, high speed, contact. Penalize slow players hard.
#   2: 2-hole    — best overall hitter. High OPS, good OBP, any speed.
#   3: 3-hole    — strong avg + power. High SLG, high OBP.
#   4: Cleanup   — maximum power. High ISO, high SLG.
#   5-6: Middle  — solid OPS, some power.
#   7-8: Bottom  — below-average contributors, slot by OPS.
#   9: 9th       — weakest hitter OR secondary table-setter if contact-only.
# ─────────────────────────────────────────────────────────────────────────────

def _archetype_score(player, slot: int, location: str, pitcher_hand: str) -> float:
    """
    Score a player for a specific batting slot (1–9).
    Higher score = better fit for that slot.
    player is a PlayerProfile; underlying Player is at player.player.
    """
    from speed import fps_to_percentile, MLB_AVG_SPEED_FPS

    key     = f"{location}_vs_{pitcher_hand}"
    probs   = getattr(player, key)
    obp     = probs.OBP
    slg     = probs.SLG
    iso     = probs.ISO
    ops     = probs.OPS
    spd     = player.player.speed_fps
    spd_pct = fps_to_percentile(spd) / 100.0    # 0–1 scale
    k_rate  = probs.event_K

    # Slowness penalty: how far below MLB average the runner is
    slow_penalty = max(0.0, (MLB_AVG_SPEED_FPS - spd) / MLB_AVG_SPEED_FPS) * 0.25

    if slot == 1:
        # Leadoff: want OBP above all, speed, contact; penalize slow and low-OBP
        return (obp * 2.0) + (spd_pct * 0.8) - slow_penalty - (k_rate * 0.5)

    elif slot == 2:
        # 2-hole: best overall hitter (most PA after #1)
        return ops * 1.5 + obp * 0.5

    elif slot == 3:
        # 3-hole: high average + good power
        return (obp * 1.2) + (slg * 1.2)

    elif slot == 4:
        # Cleanup: maximize run production; ISO and SLG dominate
        return (iso * 2.0) + (slg * 1.0)

    elif slot in (5, 6):
        # Middle order: solid OPS + some power
        return ops + iso * 0.5

    elif slot in (7, 8):
        # Bottom order: pure OPS
        return ops * 0.8 - slow_penalty * 0.5

    else:  # slot 9
        # 9th: weakest hitter; slight contact bonus (gets on before top of order)
        return ops * 0.7 + obp * 0.3


def archetype_seed(lineup: list, location: str, pitcher_hand: str) -> list:
    """
    Arrange a 9-player lineup into a batting order based on archetype scores.
    Uses a greedy assignment: for each slot 1–9, pick the unassigned player
    with the highest archetype score for that slot.
    """
    n = len(lineup)
    used = [False] * n
    order = []

    for slot in range(1, n + 1):
        best_idx   = None
        best_score = -999.0
        for i, player in enumerate(lineup):
            if not used[i]:
                score = _archetype_score(player, slot, location, pitcher_hand)
                if score > best_score:
                    best_score = score
                    best_idx   = i
        used[best_idx] = True
        order.append(lineup[best_idx])

    return order


# ─────────────────────────────────────────────────────────────────────────────
# LINEUP KEY AND DIVERSITY
# ─────────────────────────────────────────────────────────────────────────────

def lineup_key(lineup: list) -> tuple:
    return tuple(p.name for p in lineup)


def diversity_score(lineup_a: list, lineup_b: list) -> int:
    return sum(a.name != b.name for a, b in zip(lineup_a, lineup_b))


# ─────────────────────────────────────────────────────────────────────────────
# OPTIMIZER
# ─────────────────────────────────────────────────────────────────────────────

class LineupOptimizer:
    """
    Hill-climbing optimizer with:
      - Positional constraint validation (any trial lineup must be valid)
      - Archetype seeding (sane starting order instead of OPS-sort)
      - Roster swaps for players not locked to the starting 9
      - Full candidate tracking for top-N re-ranking
    """

    def __init__(
        self,
        simulator:        GameSimulator,
        n_swaps:          int   = 500,
        n_eval_games:     int   = 600,
        n_final_games:    int   = 2000,
        top_n:            int   = 10,
        accept_threshold: float = 0.018,
        seed:             int   = 42,
    ):
        self.sim              = simulator
        self.n_swaps          = n_swaps
        self.n_eval_games     = n_eval_games
        self.n_final_games    = n_final_games
        self.top_n            = top_n
        self.accept_threshold = accept_threshold
        self.seed             = seed

    def _score(self, lineup: list) -> float:
        return self.sim.run(lineup, self.n_eval_games)["mean"]

    def optimize(self, players: list) -> list:
        random.seed(self.seed)
        np.random.seed(self.seed)

        location     = self.sim.location
        pitcher_hand = getattr(self.sim, "pitcher_hand", "R")
        label        = f"{location} vs {pitcher_hand}"

        # Separate locked and unlocked players
        locked   = [p for p in players if p.player.locked]
        unlocked = [p for p in players if not p.player.locked]

        # Build a valid starting 9: all locked players + fill from unlocked
        n = 9
        start_9 = list(locked)
        for p in unlocked:
            if len(start_9) >= n:
                break
            # Only add if the resulting lineup stays potentially valid
            start_9.append(p)

        if len(start_9) < n:
            # Shouldn't happen with a properly defined roster, but guard anyway
            raise ValueError(f"Not enough players to field 9: have {len(start_9)}")

        # Trim to 9 if we somehow have more
        start_9 = start_9[:n]

        if not is_valid_lineup(start_9):
            # Try each unlocked player as a replacement until we find a valid 9
            fixed = False
            for candidate in unlocked:
                if candidate in start_9:
                    continue
                for i, p in enumerate(start_9):
                    if p.player.locked:
                        continue
                    trial = start_9[:]
                    trial[i] = candidate
                    if is_valid_lineup(trial):
                        print(f"  ⚠ Auto-fixed lineup: swapped {p.name} → {candidate.name} "
                              f"to cover missing position")
                        start_9 = trial
                        fixed = True
                        break
                if fixed:
                    break

            if not is_valid_lineup(start_9):
                print(f"  ⚠ WARNING: Could not build a positionally valid starting 9 "
                      f"for {label}. Running with best available lineup — "
                      f"results may be less meaningful.")
                print(f"  Players: {[p.name for p in start_9]}")

        # Bench = unlocked players not in start_9
        bench = [p for p in unlocked if p not in start_9]

        # Seed the batting order using archetypes
        best_lineup = archetype_seed(start_9, location, pitcher_hand)

        all_candidates: dict = {}

        def evaluate(lu: list) -> float:
            k = lineup_key(lu)
            if k not in all_candidates:
                s = self._score(lu)
                all_candidates[k] = (lu[:], s)
            return all_candidates[k][1]

        best_score = evaluate(best_lineup)
        print(f"  Archetype-seeded baseline ({label}): {best_score:.3f} R/G")
        print(f"  Starting 9: {' | '.join(p.name.split()[-1] for p in best_lineup)}")

        for _ in tqdm(range(self.n_swaps), desc=f"  Optimizing ({label})", unit="swap"):
            trial = best_lineup[:]

            roll = random.random()

            if bench and roll < 0.30:
                # Roster swap: replace one unlocked active player with a bench player
                # Never replace a locked player
                unlocked_slots = [i for i, p in enumerate(trial) if not p.player.locked]
                if not unlocked_slots:
                    continue
                active_slot  = random.choice(unlocked_slots)
                bench_player = random.choice(bench)
                trial[active_slot] = bench_player

                # Validate positional coverage after the swap
                if not is_valid_lineup(trial):
                    continue

            else:
                # Slot swap: reorder two positions
                i1, i2 = random.sample(range(n), 2)
                trial[i1], trial[i2] = trial[i2], trial[i1]

            s = evaluate(trial)
            if s > best_score + self.accept_threshold:
                best_score  = s
                best_lineup = trial
                # Update bench: the swapped-out player goes to bench, swapped-in comes off
                if bench and roll < 0.30:
                    old_player = best_lineup[active_slot]
                    # bench now contains the player that was removed
                    # (Python list is mutable; rebuild to be safe)
                    bench = [p for p in unlocked if p not in best_lineup]

        print(f"  Evaluated {len(all_candidates)} unique lineups during search")

        # ── Phase 2: re-rank top candidates at high precision ────────────────
        sorted_cands = sorted(
            all_candidates.values(), key=lambda x: x[1], reverse=True
        )[: self.top_n * 3]

        print(f"  Re-evaluating top {len(sorted_cands)} at {self.n_final_games} games…")
        ranked = []
        for lineup, _ in tqdm(sorted_cands, desc="  Re-ranking", unit="lineup"):
            sim_result = self.sim.run(lineup, self.n_final_games)
            ranked.append((lineup, sim_result))

        ranked.sort(key=lambda x: x[1]["mean"], reverse=True)
        return ranked[: self.top_n]


# ─────────────────────────────────────────────────────────────────────────────
# SLOT FREQUENCY ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def slot_frequency(ranked: list) -> dict:
    freq = defaultdict(lambda: defaultdict(int))
    for lineup, _ in ranked:
        for slot, player in enumerate(lineup, 1):
            freq[player.name][slot] += 1
    return dict(freq)
