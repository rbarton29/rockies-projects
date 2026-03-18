"""
simulator.py — Monte Carlo game simulation engine.

    FixedPitcherSimulator locks ONLY the starter's hand to the split's hand.
    Bridge and closer draw from the realistic mixed pools (~75% RHP).
    Each base slot tracks the runner's sprint speed (ft/sec), and
    advancement probability is computed per-runner using speed.advancement_probs().
    GameState.bases now stores: None (empty) or float (runner's speed_fps).
        
        Effect:
        McCarthy (29.9 ft/sec): ~43% more likely to advance extra base
        Goodman  (24.5 ft/sec): ~35% less likely to advance extra base
"""

import math
import random
from dataclasses import dataclass, field

import numpy as np

from stats import ProbSet, PlayerProfile
from speed import (
    advancement_probs,
    MLB_AVG_SPEED_FPS,
)


# ─────────────────────────────────────────────────────────────────────────────
# PITCHER MODEL
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Pitcher:
    hand:  str
    stuff: float
    role:  str


_STARTER_POOL = [
    Pitcher("R", 1.00, "starter"),
    Pitcher("R", 1.10, "starter"),
    Pitcher("R", 0.90, "starter"),
    Pitcher("L", 1.00, "starter"),
    Pitcher("L", 1.05, "starter"),
]
_STARTER_W = [0.30, 0.20, 0.20, 0.20, 0.10]

_BRIDGE_POOL = [
    Pitcher("R", 0.95, "bridge"),
    Pitcher("R", 1.05, "bridge"),
    Pitcher("L", 1.00, "bridge"),
]
_BRIDGE_W = [0.40, 0.35, 0.25]

_CLOSER_POOL = [
    Pitcher("R", 1.15, "closer"),
    Pitcher("R", 1.05, "closer"),
    Pitcher("L", 1.10, "closer"),
]
_CLOSER_W = [0.55, 0.30, 0.15]

# TTO: batter gains as pitcher degrades each time through the order
TTO_BATTER_MULT = {1: 1.00, 2: 1.06, 3: 1.12, 4: 1.15}
TTO_DECAY       = {1: 1.00, 2: 0.92, 3: 0.84, 4: 0.78}


def _sample_pitcher(pool, weights) -> Pitcher:
    return random.choices(pool, weights=weights)[0]


def _effective_stuff(pitcher: Pitcher, tto: int) -> float:
    if pitcher.role == "starter":
        return pitcher.stuff * TTO_DECAY.get(min(tto, 4), 0.78)
    return pitcher.stuff


# ─────────────────────────────────────────────────────────────────────────────
# BATTED BALL RESOLVER
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_bip(gb_pct: float, fb_pct: float, ld_pct: float,
                 bases: list, outs: int) -> tuple:
    """
    Sample a batted-ball-in-play outcome.
    Returns (event_type: str, runs_from_tag: int, extra_outs: int)

    IFFB (Infield Fly Ball): ~15% of all fly balls.
    These are soft pop-ups caught by infielders — automatic outs with zero
    runner advancement possible. They are carved out of the flyball pool first.
    """
    iffb_rate = fb_pct * 0.15
    true_fb   = fb_pct * 0.85
    total     = gb_pct + true_fb + ld_pct

    if total <= 0:
        return ("groundout", 0, 1)

    r = random.random()

    if r < iffb_rate:
        return ("popup", 0, 1)
    r -= iffb_rate

    remaining = 1.0 - iffb_rate

    if r < (gb_pct / total) * remaining:
        # GIDP: runner on 1st, fewer than 2 outs, ~10% of grounders
        if bases[0] is not None and outs < 2 and random.random() < 0.10:
            return ("gidp", 0, 2)
        return ("groundout", 0, 1)
    r -= (gb_pct / total) * remaining

    if r < (true_fb / total) * remaining:
        # Tag-up: runner on 3rd scores ~75% of the time on a flyout
        if bases[2] is not None and random.random() < 0.75:
            return ("flyout_score", 1, 1)
        return ("flyout", 0, 1)

    return ("lineout", 0, 1)


# ─────────────────────────────────────────────────────────────────────────────
# PROBS ADJUSTER
# ─────────────────────────────────────────────────────────────────────────────

def _adjust_probs(probs: ProbSet, tto: int, pitcher_stuff: float) -> ProbSet:
    tto_mult = TTO_BATTER_MULT.get(min(tto, 4), 1.15)
    net      = tto_mult / max(pitcher_stuff, 0.01)
    offensive = ("event_HR","event_3B","event_2B","event_1B","event_BB","event_HBP","event_bip")
    adjusted  = ProbSet(**probs.__dict__)
    for ev in offensive:
        setattr(adjusted, ev, getattr(probs, ev) * net)
    return adjusted.normalized()


# ─────────────────────────────────────────────────────────────────────────────
# PLATE APPEARANCE SIMULATOR
# ─────────────────────────────────────────────────────────────────────────────

def _sim_pa(
    probs:         ProbSet,
    gb_pct:        float,
    fb_pct:        float,
    ld_pct:        float,
    bases:         list,     # [speed_fps_or_None, speed_fps_or_None, speed_fps_or_None]
    outs:          int,
    location:      str,      # 'home' or 'away' — affects Coors advancement boost
    batter_speed:  float,    # batter's sprint speed, placed on base if they reach
) -> tuple:
    """
    Simulate one plate appearance with speed-based runner advancement.

    bases: list of [1B, 2B, 3B] where each slot is either:
      None  — base is empty
      float — base is occupied; value is the runner's sprint speed in ft/sec

    Single advancement rules:
      Runner on 3rd: always scores (90 ft, easy; no speed gate needed)
      Runner on 2nd: scores with probability from advancement_probs(runner_speed, location)
                     If they don't score, they advance to 3rd
      Runner on 1st: advances to 3rd with probability from advancement_probs(runner_speed, location)
                     If they don't reach 3rd, they advance to 2nd (standard advance)

    Double advancement rules (deterministic — doubles are slow enough that runners
    on 2nd and 3rd always score, and runner on 1st always reaches 3rd):
      Runner on 3rd: always scores
      Runner on 2nd: always scores
      Runner on 1st: always advances to 3rd

    Returns (runs_scored: int, outs_added: int, new_bases: list)
    """
    bases     = list(bases)
    runs      = 0
    outs_add  = 0

    r   = random.random()
    cum = 0.0

    for event_key in ProbSet.EVENT_KEYS:
        cum += getattr(probs, event_key)
        if r >= cum:
            continue

        # ── Home run ─────────────────────────────────────────────────────────
        if event_key == "event_HR":
            runs += sum(1 for b in bases if b is not None) + 1
            bases = [None, None, None]

        # ── Triple ───────────────────────────────────────────────────────────
        elif event_key == "event_3B":
            runs += sum(1 for b in bases if b is not None)
            bases = [None, None, batter_speed]

        # ── Double ───────────────────────────────────────────────────────────
        elif event_key == "event_2B":
            # Runner on 3rd scores
            if bases[2] is not None:
                runs += 1

            # Runner on 2nd scores
            if bases[1] is not None:
                runs += 1

            # Runner on 1st advances to 3rd
            runner_on_first = bases[0]

            bases = [None, None, None]
            bases[1] = batter_speed

            if runner_on_first is not None:
                bases[2] = runner_on_first

        # ── Single ───────────────────────────────────────────────────────────
        elif event_key == "event_1B":
            # Runner on 3rd always scores
            if bases[2] is not None:
                runs += 1
                bases[2] = None

            # Runner on 2nd: scores or stays at 3rd (speed-based)
            if bases[1] is not None:
                runner_speed = bases[1]
                p_score, _ = advancement_probs(runner_speed, location)
                if random.random() < p_score:
                    runs += 1
                    bases[1] = None
                else:
                    # Runner stays — moves to 3rd only if 3rd is empty
                    if bases[2] is None:
                        bases[2] = runner_speed
                    bases[1] = None

            # Runner on 1st: may advance to 3rd (speed-based)
            if bases[0] is not None:
                runner_speed = bases[0]
                _, p_1st_to_3rd = advancement_probs(runner_speed, location)
                if random.random() < p_1st_to_3rd and bases[2] is None:
                    bases[2] = runner_speed
                elif bases[1] is None:
                    bases[1] = runner_speed
                else:
                    # Both 2nd and 3rd occupied — stay at 2nd if possible
                    if bases[1] is None:
                        bases[1] = runner_speed
                bases[0] = None

            # Batter takes first
            bases[0] = batter_speed

        # ── Walk or HBP ──────────────────────────────────────────────────────
        elif event_key in ("event_BB", "event_HBP"):
            # Force-advance if bases loaded
            if bases[0] is not None and bases[1] is not None and bases[2] is not None:
                runs += 1
            elif bases[0] is not None and bases[1] is not None:
                bases[2] = bases[1]
            elif bases[0] is not None:
                bases[1] = bases[0]
            bases[0] = batter_speed

        # ── Strikeout ─────────────────────────────────────────────────────────
        elif event_key == "event_K":
            outs_add += 1

        # ── Sac fly ───────────────────────────────────────────────────────────
        elif event_key == "event_sac_fly":
            if bases[2] is not None:
                runs += 1
                bases[2] = None
            outs_add += 1

        # ── Sac bunt ──────────────────────────────────────────────────────────
        elif event_key == "event_sac_bunt":
            if bases[2] is not None:
                runs += 1
            bases[2] = bases[1]
            bases[1] = bases[0]
            bases[0] = None
            outs_add += 1

        # ── Batted ball in play ────────────────────────────────────────────────
        elif event_key == "event_bip":
            bip_type, tag_runs, extra_outs = _resolve_bip(gb_pct, fb_pct, ld_pct, bases, outs)

            if bip_type == "gidp":
                outs_add += extra_outs
                bases[0] = None

            elif bip_type in ("groundout", "lineout", "popup"):
                # No runner advancement on groundout, lineout, or infield popup
                outs_add += 1

            elif bip_type in ("flyout", "flyout_score"):
                outs_add += 1
                if bip_type == "flyout_score" and bases[2] is not None:
                    runs += 1
                    bases[2] = None

        break  # only one event per PA

    return runs, outs_add, bases


# ─────────────────────────────────────────────────────────────────────────────
# GAME SIMULATOR
# ─────────────────────────────────────────────────────────────────────────────

class GameSimulator:
    """
    Simulates 9-inning games for a lineup and game location.

    Pitcher model:
      Innings 1–5:  starting pitcher (with TTO penalty applied)
      Innings 6–7:  bridge/setup reliever (fresh, no TTO)
      Innings 8–9:  closer (fresh, no TTO)

    All four location×pitcher_hand splits are used automatically based on
    which pitcher is currently pitching.
    """

    def __init__(self, location: str):
        assert location in ("home", "away")
        self.location = location

    def _draw_pitchers(self) -> tuple:
        """Draw a starter, bridge, and closer for this game."""
        return (
            _sample_pitcher(_STARTER_POOL, _STARTER_W),
            _sample_pitcher(_BRIDGE_POOL,  _BRIDGE_W),
            _sample_pitcher(_CLOSER_POOL,  _CLOSER_W),
        )

    def simulate_game(self, lineup: list) -> float:
        starter, bridge, closer = self._draw_pitchers()

        total_runs  = 0.0
        batter_idx  = 0
        pa_per_slot = [0] * len(lineup)

        for inning in range(1, 10):
            pitcher = starter if inning <= 5 else (bridge if inning <= 7 else closer)

            outs  = 0
            bases = [None, None, None]   # each slot: None or runner speed_fps

            while outs < 3:
                pos    = batter_idx % len(lineup)
                player = lineup[pos]
                pa_per_slot[pos] += 1

                tto   = min(pa_per_slot[pos], 4) if pitcher.role == "starter" else 1
                stuff = _effective_stuff(pitcher, tto)

                raw_probs = player.get_probs(self.location, pitcher.hand)
                adj_probs = _adjust_probs(raw_probs, tto, stuff)

                runs_scored, outs_added, bases = _sim_pa(
                    adj_probs,
                    player.gb_pct,
                    player.fb_pct,
                    player.ld_pct,
                    bases,
                    outs,
                    self.location,
                    player.player.speed_fps,
                )
                total_runs += runs_scored
                outs       += outs_added
                batter_idx += 1

        return total_runs

    def run(self, lineup: list, n_games: int) -> dict:
        results = [self.simulate_game(lineup) for _ in range(n_games)]
        arr     = np.array(results, dtype=float)
        hist    = [0] * 16
        for r in results:
            hist[min(int(r), 15)] += 1
        return {
            "mean":    float(arr.mean()),
            "std":     float(arr.std()),
            "median":  float(np.median(arr)),
            "p10":     float(np.percentile(arr, 10)),
            "p90":     float(np.percentile(arr, 90)),
            "hist":    hist,
            "n":       n_games,
            "location": self.location,
        }


# ─────────────────────────────────────────────────────────────────────────────
# FIXED-PITCHER-HAND SIMULATOR
# For the 4 splits: Home vs RHP / Home vs LHP / Away vs RHP / Away vs LHP
# ─────────────────────────────────────────────────────────────────────────────

class FixedPitcherSimulator(GameSimulator):
    """
    Locks only the STARTER's hand to a fixed value for the split.
    Bridge and closer draw from realistic mixed pools.
    - "Home vs LHP" means the opposing STARTER is a left-hander
    - The 7th-inning setup man and closer are whoever the opponent rolls out,
      which is ~75% RHP regardless of the starter's hand
    """

    def __init__(self, location: str, pitcher_hand: str):
        super().__init__(location)
        assert pitcher_hand in ("L", "R")
        self.pitcher_hand = pitcher_hand

    def _draw_pitchers(self) -> tuple:
        # Starter: hand locked to split, stuff drawn from normal distribution
        starter_template = _sample_pitcher(_STARTER_POOL, _STARTER_W)
        starter = Pitcher(self.pitcher_hand, starter_template.stuff, "starter")

        # Bridge and closer: fully from mixed pools (no hand constraint)
        bridge  = _sample_pitcher(_BRIDGE_POOL,  _BRIDGE_W)
        closer  = _sample_pitcher(_CLOSER_POOL,  _CLOSER_W)

        return starter, bridge, closer
