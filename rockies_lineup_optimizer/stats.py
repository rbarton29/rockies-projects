"""
stats.py — Convert RawStats into per-PA event probability sets.

Derived metrics added to ProbSet:
  ISO   = SLG - AVG          (isolated power; strips out singles)
  PWR   = XBH / H            (power factor; XBH tendency independent of avg)
  GB_FB = gb_pct / fb_pct    (groundball-to-flyball ratio; ~1.2 is MLB avg)
  
"""

from dataclasses import dataclass, field
from roster import RawStats, Player


# ─────────────────────────────────────────────────────────────────────────────
# PROBABILITY SET
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ProbSet:
    """
    Per-PA event probabilities for one split (location × pitcher hand).
    All event_* fields sum to 1.0.
    """
    # Outcome probabilities
    event_HR:       float = 0.0
    event_3B:       float = 0.0
    event_2B:       float = 0.0
    event_1B:       float = 0.0
    event_BB:       float = 0.0
    event_HBP:      float = 0.0
    event_K:        float = 0.0
    event_sac_fly:  float = 0.0
    event_sac_bunt: float = 0.0
    event_bip:      float = 0.0   # batted ball in play (not SF/SH) — broken down in simulator

    # Descriptive / display stats
    AVG:  float = 0.0
    OBP:  float = 0.0
    SLG:  float = 0.0
    OPS:  float = 0.0
    ISO:  float = 0.0   # SLG - AVG
    PWR:  float = 0.0   # XBH / H  (power factor)
    PA:   int   = 0
    HR_count: int = 0
    SB:   int   = 0

    EVENT_KEYS = (
        "event_HR", "event_3B", "event_2B", "event_1B",
        "event_BB", "event_HBP", "event_K",
        "event_sac_fly", "event_sac_bunt", "event_bip",
    )

    def normalized(self) -> "ProbSet":
        """Return a copy with event probabilities normalized to sum to 1."""
        total = sum(getattr(self, k) for k in self.EVENT_KEYS)
        if total <= 0:
            return self
        out = ProbSet(**self.__dict__)
        for k in self.EVENT_KEYS:
            setattr(out, k, getattr(self, k) / total)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# LEAGUE AVERAGE FALLBACK
# ─────────────────────────────────────────────────────────────────────────────

_LEAGUE_AVG_RAW = RawStats(
    PA=550, AB=490, H=125, B2=27, B3=3, HR=18,
    BB=52, HBP=7, SF=5, SH=2, K=130, SB=8,
)


def _league_avg_probs() -> ProbSet:
    return compute_probs(_LEAGUE_AVG_RAW)


# ─────────────────────────────────────────────────────────────────────────────
# CORE CONVERSION
# ─────────────────────────────────────────────────────────────────────────────

def compute_probs(raw: RawStats) -> ProbSet:
    """
    Convert a RawStats counting block into a normalized ProbSet.

    Event hierarchy (in sampling order):
      HR → 3B → 2B → 1B → BB → HBP → K → sac_fly → sac_bunt → bip (field out / hit by BIP)

    ISO  = SLG - AVG  (standard sabermetric isolated power definition)
    PWR  = (2B + 3B + HR) / H  (fraction of hits going for extra bases)
    GB/FB is stored at the Player level, not per-ProbSet.
    """
    if raw.PA < 10:
        return _league_avg_probs()

    pa  = raw.PA
    ab  = raw.AB
    h   = raw.H
    b1  = max(h - raw.B2 - raw.B3 - raw.HR, 0)  # singles

    # Per-PA event rates
    r_hr  = raw.HR  / pa
    r_3b  = raw.B3  / pa
    r_2b  = raw.B2  / pa
    r_1b  = b1      / pa
    r_bb  = raw.BB  / pa
    r_hbp = raw.HBP / pa
    r_k   = raw.K   / pa
    r_sf  = raw.SF  / pa
    r_sh  = raw.SH  / pa

    # BIP = everything not accounted for above (GB/FB/LD outs, errors, etc.)
    r_bip = max(
        1.0 - r_hr - r_3b - r_2b - r_1b - r_bb - r_hbp - r_k - r_sf - r_sh,
        0.01,
    )

    # Descriptive stats
    avg = h / ab if ab > 0 else 0.0
    denom_obp = pa - raw.SF - raw.SH
    obp = (h + raw.BB + raw.HBP) / denom_obp if denom_obp > 0 else 0.0
    tb  = b1 + 2 * raw.B2 + 3 * raw.B3 + 4 * raw.HR
    slg = tb / ab if ab > 0 else 0.0

    iso = slg - avg
    pwr = (raw.B2 + raw.B3 + raw.HR) / h if h > 0 else 0.0

    prob = ProbSet(
        event_HR       = r_hr,
        event_3B       = r_3b,
        event_2B       = r_2b,
        event_1B       = r_1b,
        event_BB       = r_bb,
        event_HBP      = r_hbp,
        event_K        = r_k,
        event_sac_fly  = r_sf,
        event_sac_bunt = r_sh,
        event_bip      = r_bip,
        AVG      = avg,
        OBP      = obp,
        SLG      = slg,
        OPS      = obp + slg,
        ISO      = iso,
        PWR      = pwr,
        PA       = pa,
        HR_count = raw.HR,
        SB       = raw.SB,
    )
    return prob.normalized()


# ─────────────────────────────────────────────────────────────────────────────
# PLAYER PROBABILITY PROFILE
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PlayerProfile:
    """
    A Player annotated with all four ProbSet splits, ready for simulation.
    Splits:
      home_vs_R  — Coors Field vs right-handed pitcher
      home_vs_L  — Coors Field vs left-handed pitcher
      away_vs_R  — road game vs right-handed pitcher
      away_vs_L  — road game vs left-handed pitcher
    """
    player:     Player
    home_vs_R:  ProbSet
    home_vs_L:  ProbSet
    away_vs_R:  ProbSet
    away_vs_L:  ProbSet

    # Convenience pass-throughs
    @property
    def name(self) -> str:
        return self.player.name

    @property
    def pos(self) -> str:
        return self.player.pos

    @property
    def bats(self) -> str:
        return self.player.bats

    @property
    def games(self) -> int:
        return self.player.games

    @property
    def gb_pct(self) -> float:
        return self.player.gb_pct

    @property
    def fb_pct(self) -> float:
        return self.player.fb_pct

    @property
    def ld_pct(self) -> float:
        return self.player.ld_pct

    @property
    def notes(self) -> str:
        return self.player.notes

    @property
    def verified_splits(self) -> bool:
        return self.player.verified_splits

    def get_probs(self, location: str, pitcher_hand: str) -> ProbSet:
        """
        Select the correct ProbSet for a given game context.
        location:     'home' or 'away'
        pitcher_hand: 'L'  or 'R'
        """
        key = f"{location}_vs_{pitcher_hand}"
        return getattr(self, key)

    def gb_fb_ratio(self) -> float:
        return self.gb_pct / max(self.fb_pct, 0.01)


def build_profiles(roster: list) -> list:
    """Convert a list of Player objects into PlayerProfile objects."""
    profiles = []
    for player in roster:
        profiles.append(PlayerProfile(
            player    = player,
            home_vs_R = compute_probs(player.home_vs_R),
            home_vs_L = compute_probs(player.home_vs_L),
            away_vs_R = compute_probs(player.away_vs_R),
            away_vs_L = compute_probs(player.away_vs_L),
        ))
    return profiles
