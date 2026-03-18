"""
roster.py — Player data classes and 2025/2026 Rockies roster.

CALIBRATION PHILOSOPHY:
  Albuquerque Isotopes park sits at 5,312 ft — virtually identical to Coors Field
  (5,280 ft). Karros's AAA HOME stats are therefore already elevation-calibrated.
  We apply only a small park-geometry correction (ABQ is smaller, friendlier to
  HR; Coors is larger, friendlier to triples) rather than a full altitude boost.

  Rumfield's entire minor-league career was at sea level (SWB, Somerset, etc.).
  His stats require two sequential adjustments:
    1. apply_aaa_to_mlb_discount()  — quality of competition
    2. calibrate_home_from_away()   — Coors park factors

  MLB translation discount factors (from standard projection systems, ZiPS/Steamer):
    H:   × 0.89    OBP boost from walks: × 0.91    HR: × 0.87
    2B:  × 0.88    3B: × 0.85              K: × 1.02
  These are averages; individual variance is high for short samples.

  AA-to-MLB discount is slightly larger:
    H:   × 0.85    BB: × 0.88    HR: × 0.82    2B: × 0.84
    
"""

from dataclasses import dataclass, field
from speed import get_speed


# ─────────────────────────────────────────────────────────────────────────────
# POSITION ELIGIBILITY
# ─────────────────────────────────────────────────────────────────────────────

# Standard MLB positional eligibility mapping.
# Used by the optimizer to verify that any 9-man lineup covers all required slots.
POSITION_SLOTS = {"C", "1B", "2B", "3B", "SS", "LF", "CF", "RF", "DH"}

# Which players can cover which defensive positions in a pinch
_ELIGIBLE: dict = {}   # populated in get_roster_2026()


# ─────────────────────────────────────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RawStats:
    PA:  int = 0;  AB:  int = 0;  H:   int = 0
    B2:  int = 0;  B3:  int = 0;  HR:  int = 0
    BB:  int = 0;  HBP: int = 0;  SF:  int = 0
    SH:  int = 0;  K:   int = 0;  SB:  int = 0

    @classmethod
    def from_dict(cls, d: dict) -> "RawStats":
        return cls(
            PA=d.get("PA",0), AB=d.get("AB",0), H=d.get("H",0),
            B2=d.get("2B",0), B3=d.get("3B",0), HR=d.get("HR",0),
            BB=d.get("BB",0), HBP=d.get("HBP",0), SF=d.get("SF",0),
            SH=d.get("SH",0), K=d.get("K",0),   SB=d.get("SB",0),
        )

    def scale(self, f: float) -> "RawStats":
        return RawStats(
            PA=int(self.PA*f), AB=int(self.AB*f), H=int(self.H*f),
            B2=int(self.B2*f), B3=int(self.B3*f), HR=int(self.HR*f),
            BB=int(self.BB*f), HBP=int(self.HBP*f), SF=int(self.SF*f),
            SH=int(self.SH*f), K=int(self.K*f),   SB=int(self.SB*f),
        )

    def add(self, o: "RawStats") -> "RawStats":
        return RawStats(
            PA=self.PA+o.PA, AB=self.AB+o.AB, H=self.H+o.H,
            B2=self.B2+o.B2, B3=self.B3+o.B3, HR=self.HR+o.HR,
            BB=self.BB+o.BB, HBP=self.HBP+o.HBP, SF=self.SF+o.SF,
            SH=self.SH+o.SH, K=self.K+o.K,   SB=self.SB+o.SB,
        )


@dataclass
class Player:
    name:            str
    pos:             str            # primary position label (display only)
    eligible_pos:    list           # list of positions this player can cover
    bats:            str            # L / R / S
    games:           int
    home_vs_R:       RawStats
    home_vs_L:       RawStats
    away_vs_R:       RawStats
    away_vs_L:       RawStats
    gb_pct:          float = 0.42
    fb_pct:          float = 0.35
    ld_pct:          float = 0.23
    speed_fps:       float = 27.0
    locked:          bool  = False  # True = always in lineup (not subject to bench-swap)
    verified_splits: bool  = False
    notes:           str   = ""


# ─────────────────────────────────────────────────────────────────────────────
# MINOR-LEAGUE TO MLB QUALITY DISCOUNTS
#
# These multipliers adjust counting stats to reflect the quality gap between
# minor-league pitching and MLB pitching. Applied to all offensive events.
# Sources: ZiPS, Steamer, and BP's PECOTA translation coefficients.
#
# Note on Albuquerque home stats: we do NOT apply the full AAA discount to
# Karros's Albuquerque home stats for the home-projection path, only the
# quality adjustment. The altitude effect is already baked in.
# ─────────────────────────────────────────────────────────────────────────────

def apply_aaa_to_mlb_discount(stats: RawStats) -> RawStats:
    """
    Apply quality-of-competition discount for AAA → MLB translation.
    Does NOT include a park factor — apply separately for home/away.
    """
    if stats.PA < 10:
        return stats
    return RawStats(
        PA=stats.PA,
        AB=stats.AB,
        H=int(stats.H   * 0.89),
        B2=int(stats.B2 * 0.88),
        B3=int(stats.B3 * 0.85),
        HR=int(stats.HR * 0.87),
        BB=int(stats.BB * 0.91),
        HBP=stats.HBP,
        SF=stats.SF,
        SH=stats.SH,
        K=int(stats.K   * 1.02),
        SB=int(stats.SB * 0.85),
    )


def apply_aa_to_mlb_discount(stats: RawStats) -> RawStats:
    """
    Slightly larger discount for Double-A → MLB translation.
    Double-A pitching is meaningfully worse than AAA.
    """
    if stats.PA < 10:
        return stats
    return RawStats(
        PA=stats.PA,
        AB=stats.AB,
        H=int(stats.H   * 0.85),
        B2=int(stats.B2 * 0.84),
        B3=int(stats.B3 * 0.82),
        HR=int(stats.HR * 0.82),
        BB=int(stats.BB * 0.88),
        HBP=stats.HBP,
        SF=stats.SF,
        SH=stats.SH,
        K=int(stats.K   * 1.04),
        SB=int(stats.SB * 0.83),
    )


def apply_abq_to_coors_correction(stats: RawStats) -> RawStats:
    """
    Small park-geometry correction from Isotopes Park (ABQ) to Coors Field.

    Both parks are at ~5,300 ft altitude so the thin-air effects are identical.
    The geometry difference:
      ABQ dimensions: 340 LF / 400 CF / 328 RF  (more compact → more HR)
      Coors dimensions: 347 LF / 415 CF / 350 RF  (larger → more triples, fewer HR)

    Net adjustments: slight HR penalty (×0.97), triples bonus (×1.08),
    doubles bonus (×1.03 — more gap room at Coors). All other events unchanged.
    """
    if stats.PA < 10:
        return stats
    return RawStats(
        PA=stats.PA,   AB=stats.AB,   H=stats.H,
        B2=int(stats.B2 * 1.03),
        B3=int(stats.B3 * 1.08),
        HR=int(stats.HR * 0.97),
        BB=stats.BB,   HBP=stats.HBP,  SF=stats.SF,
        SH=stats.SH,   K=stats.K,      SB=stats.SB,
    )


# ─────────────────────────────────────────────────────────────────────────────
# COORS PARK FACTOR (for sea-level stats → Coors home projection)
# ─────────────────────────────────────────────────────────────────────────────

COORS_PARK_FACTORS = {
    "H_single": 1.06, "2B": 1.12, "3B": 1.55,
    "HR": 1.12, "BB": 1.03, "K": 0.97,
}

def calibrate_home_from_away(away: RawStats) -> RawStats:
    """Derive Coors home stats from neutral/sea-level away stats."""
    if away.PA < 10:
        return away
    singles = max(away.H - away.B2 - away.B3 - away.HR, 0)
    h_b2 = int(away.B2 * COORS_PARK_FACTORS["2B"])
    h_b3 = int(away.B3 * COORS_PARK_FACTORS["3B"])
    h_hr = int(away.HR * COORS_PARK_FACTORS["HR"])
    h_bb = int(away.BB * COORS_PARK_FACTORS["BB"])
    h_k  = int(away.K  * COORS_PARK_FACTORS["K"])
    h_s  = int(singles * COORS_PARK_FACTORS["H_single"])
    return RawStats(
        PA=away.PA, AB=int(away.AB * 1.02), H=h_s + h_b2 + h_b3 + h_hr,
        B2=h_b2, B3=h_b3, HR=h_hr, BB=h_bb,
        HBP=away.HBP, SF=away.SF, SH=away.SH,
        K=h_k, SB=int(away.SB * 1.08),
    )


# ─────────────────────────────────────────────────────────────────────────────
# BLEND HELPER
# ─────────────────────────────────────────────────────────────────────────────

def blend_seasons(seasons: list, weights: list) -> RawStats:
    """Normalize each season to 500 PA, blend by weight."""
    total_w = sum(weights)
    weights  = [w / total_w for w in weights]
    blended  = RawStats()
    for stats, w in zip(seasons, weights):
        if stats.PA < 5:
            continue
        blended = blended.add(stats.scale(500 / max(stats.PA, 1)).scale(w))
    return blended


# ─────────────────────────────────────────────────────────────────────────────
# PLATOON SPLIT HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _split_platoon(agg: RawStats, boost: float) -> tuple:
    """
    Derive vs_R and vs_L from aggregate stats + boost factor.
    Returns (vs_disadvantaged_stats, vs_advantaged_stats) at 70/30 PA split.
    Caller is responsible for assigning correctly to vs_R / vs_L based on
    the batter's handedness.
    """
    adv = 1.0 + boost
    dis = 1.0 - boost * 0.4

    def frac(raw: RawStats, pa_frac: float, hit_scale: float) -> RawStats:
        return RawStats(
            PA=max(int(raw.PA * pa_frac), 1),
            AB=max(int(raw.AB * pa_frac), 1),
            H=int(raw.H * pa_frac * hit_scale),
            B2=int(raw.B2 * pa_frac * hit_scale),
            B3=int(raw.B3 * pa_frac * hit_scale),
            HR=int(raw.HR * pa_frac * hit_scale),
            BB=int(raw.BB * pa_frac * hit_scale),
            HBP=int(raw.HBP * pa_frac),
            SF=int(raw.SF * pa_frac),
            SH=int(raw.SH * pa_frac),
            K=int(raw.K * pa_frac),
            SB=int(raw.SB * pa_frac * hit_scale),
        )

    dis_stats = frac(agg, 0.70, dis)
    adv_stats = frac(agg, 0.30, adv)
    return dis_stats, adv_stats


def _apply_platoon(name, bats, home_agg, away_agg, boost) -> tuple:
    """
    Given home/away aggregates and a boost, return (hvsR, hvsL, avsR, avsL).
    Handles handedness direction correctly:
      L batter: advantaged vs RHP  → _split_platoon returns (dis=vsL, adv=vsR); swap
      R batter: advantaged vs LHP  → _split_platoon returns (dis=vsR, adv=vsL); natural
      S batter: small boost both   → handled separately
    """
    if bats == "L":
        # L batter: adv vs RHP. _split_platoon puts adv in slot 2.
        # We want vs_R=adv, vs_L=dis, so swap the return values.
        h_vsL, h_vsR = _split_platoon(home_agg, boost)
        a_vsL, a_vsR = _split_platoon(away_agg, boost)
    elif bats == "R":
        # R batter: adv vs LHP. _split_platoon slot 2 = adv = vsL. Natural.
        h_vsR, h_vsL = _split_platoon(home_agg, boost)
        a_vsR, a_vsL = _split_platoon(away_agg, boost)
    else:
        # Switch hitter: modest boost each way
        h_vsR, h_vsL = _split_platoon(home_agg, boost * 0.5)
        a_vsR, a_vsL = _split_platoon(away_agg, boost * 0.5)
    return h_vsR, h_vsL, a_vsR, a_vsL


# ─────────────────────────────────────────────────────────────────────────────
# PLAYER BUILDER SHORTCUTS
# ─────────────────────────────────────────────────────────────────────────────

def _make(name, pos, eligible_pos, bats, games,
          home, away, gb_pct, fb_pct, ld_pct,
          platoon_boost, locked=False, verified_splits=False, notes="") -> Player:
    home_agg = RawStats.from_dict(home)
    away_agg = RawStats.from_dict(away)
    h_vsR, h_vsL, a_vsR, a_vsL = _apply_platoon(name, bats, home_agg, away_agg, platoon_boost)
    return Player(
        name=name, pos=pos, eligible_pos=eligible_pos,
        bats=bats, games=games,
        home_vs_R=h_vsR, home_vs_L=h_vsL,
        away_vs_R=a_vsR, away_vs_L=a_vsL,
        gb_pct=gb_pct, fb_pct=fb_pct, ld_pct=ld_pct,
        speed_fps=get_speed(name),
        locked=locked, verified_splits=verified_splits, notes=notes,
    )


def _make_from_away(name, pos, eligible_pos, bats, games,
                    away_dict, gb_pct, fb_pct, ld_pct,
                    platoon_boost, locked=False, notes="") -> Player:
    away_agg = RawStats.from_dict(away_dict)
    home_agg = calibrate_home_from_away(away_agg)

    def _d(s): return {"PA":s.PA,"AB":s.AB,"H":s.H,"2B":s.B2,"3B":s.B3,"HR":s.HR,
                       "BB":s.BB,"HBP":s.HBP,"SF":s.SF,"SH":s.SH,"K":s.K,"SB":s.SB}
    return _make(name, pos, eligible_pos, bats, games,
                 home=_d(home_agg), away=away_dict,
                 gb_pct=gb_pct, fb_pct=fb_pct, ld_pct=ld_pct,
                 platoon_boost=platoon_boost, locked=locked,
                 verified_splits=False, notes=notes)


# ─────────────────────────────────────────────────────────────────────────────
# 2025 ROSTER
# ─────────────────────────────────────────────────────────────────────────────

def get_roster_2025() -> list:
    return [
        _make("Hunter Goodman","C",["C"],"R",144,
            home={"PA":276,"AB":245,"H":75,"2B":15,"3B":1,"HR":13,"BB":27,"HBP":2,"SF":2,"SH":0,"K":61,"SB":3},
            away={"PA":264,"AB":238,"H":59,"2B":13,"3B":1,"HR":18,"BB":24,"HBP":1,"SF":1,"SH":0,"K":64,"SB":1},
            gb_pct=0.38,fb_pct=0.42,ld_pct=0.20,platoon_boost=0.08,locked=True,verified_splits=True,
            notes="NL Silver Slugger 2025."),
        _make("Jordan Beck","LF",["LF","RF","CF"],"R",148,
            home={"PA":302,"AB":277,"H":77,"2B":15,"3B":3,"HR":9,"BB":23,"HBP":1,"SF":1,"SH":0,"K":86,"SB":11},
            away={"PA":287,"AB":262,"H":62,"2B":12,"3B":2,"HR":7,"BB":20,"HBP":1,"SF":1,"SH":0,"K":88,"SB":8},
            gb_pct=0.44,fb_pct=0.35,ld_pct=0.21,platoon_boost=0.06,locked=True),
        _make("Mickey Moniak","RF",["RF","LF","CF"],"L",127,
            home={"PA":238,"AB":214,"H":68,"2B":13,"3B":3,"HR":13,"BB":21,"HBP":2,"SF":1,"SH":0,"K":58,"SB":4},
            away={"PA":230,"AB":210,"H":47,"2B":10,"3B":2,"HR":11,"BB":18,"HBP":1,"SF":1,"SH":0,"K":64,"SB":2},
            gb_pct=0.34,fb_pct=0.45,ld_pct=0.21,platoon_boost=0.10,locked=True,verified_splits=True),
        _make("Ezequiel Tovar","SS",["SS","2B"],"R",95,
            home={"PA":220,"AB":195,"H":66,"2B":14,"3B":1,"HR":12,"BB":21,"HBP":1,"SF":1,"SH":0,"K":49,"SB":9},
            away={"PA":183,"AB":168,"H":33,"2B":7,"3B":0,"HR":3,"BB":12,"HBP":1,"SF":0,"SH":0,"K":46,"SB":4},
            gb_pct=0.40,fb_pct=0.36,ld_pct=0.24,platoon_boost=0.07,locked=True,verified_splits=True,
            notes="⚠ Extreme Coors split. Injury risk."),
        _make("Brenton Doyle","CF",["CF","LF","RF"],"R",130,
            home={"PA":262,"AB":235,"H":72,"2B":16,"3B":3,"HR":9,"BB":22,"HBP":3,"SF":2,"SH":0,"K":70,"SB":13},
            away={"PA":249,"AB":225,"H":36,"2B":8,"3B":1,"HR":6,"BB":16,"HBP":1,"SF":1,"SH":0,"K":83,"SB":7},
            gb_pct=0.38,fb_pct=0.36,ld_pct=0.26,platoon_boost=0.05,locked=True,verified_splits=True,
            notes="⚠ .306 home vs .160 away. GG defense."),
        _make("Tyler Freeman","DH",["2B","3B","SS","DH","LF","RF"],"R",110,
            home={"PA":215,"AB":194,"H":61,"2B":12,"3B":1,"HR":1,"BB":17,"HBP":2,"SF":1,"SH":0,"K":24,"SB":10},
            away={"PA":197,"AB":183,"H":45,"2B":8,"3B":1,"HR":1,"BB":12,"HBP":1,"SF":1,"SH":0,"K":22,"SB":5},
            gb_pct=0.52,fb_pct=0.26,ld_pct=0.22,platoon_boost=0.05,
            notes="Elite contact. Classic leadoff/table-setter."),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# 2026 ROSTER — 12 candidates
#
# LINEUP STRUCTURE (8 locks + 1 competing DH slot):
#   C:   Goodman         (only catcher, must start)
#   1B:  Rumfield        (spring winner; locked)
#   2B:  Castro          (per spring plan; locked)
#   3B:  Karros          (spring winner; locked)
#   SS:  Tovar           (locked, health permitting; Ritter backup)
#   LF:  Beck            (locked)
#   CF:  Doyle           (locked, Gold Glove)
#   RF:  Moniak          (locked)
#   DH:  McCarthy / Freeman / Julien (competing for 9th spot)
#
# The optimizer selects which 9 of the 12 start and in what order.
# Positional constraints are enforced: any valid 9-man lineup must cover
# C, 1B, 2B, 3B, SS, LF, CF, RF, and DH using eligible_pos mappings.
# ─────────────────────────────────────────────────────────────────────────────

def get_roster_2026() -> list:

    # ── Locked starters ───────────────────────────────────────────────────────

    goodman = _make("Hunter Goodman","C",["C"],"R",150,
        home={"PA":276,"AB":245,"H":75,"2B":15,"3B":1,"HR":13,"BB":27,"HBP":2,"SF":2,"SH":0,"K":61,"SB":3},
        away={"PA":264,"AB":238,"H":59,"2B":13,"3B":1,"HR":18,"BB":24,"HBP":1,"SF":1,"SH":0,"K":64,"SB":1},
        gb_pct=0.38,fb_pct=0.42,ld_pct=0.20,platoon_boost=0.08,
        locked=True,verified_splits=True,
        notes="Silver Slugger. Middle-order slugger type, not a leadoff candidate.")

    beck = _make("Jordan Beck","LF",["LF","RF","CF","DH"],"R",145,
        home={"PA":302,"AB":277,"H":77,"2B":15,"3B":3,"HR":9,"BB":23,"HBP":1,"SF":1,"SH":0,"K":86,"SB":11},
        away={"PA":287,"AB":262,"H":62,"2B":12,"3B":2,"HR":7,"BB":20,"HBP":1,"SF":1,"SH":0,"K":88,"SB":8},
        gb_pct=0.44,fb_pct=0.35,ld_pct=0.21,platoon_boost=0.06,
        locked=True,notes="Year 2. High K rate concern.")

    moniak = _make("Mickey Moniak","RF",["RF","LF","CF","DH"],"L",130,
        home={"PA":238,"AB":214,"H":68,"2B":13,"3B":3,"HR":13,"BB":21,"HBP":2,"SF":1,"SH":0,"K":58,"SB":4},
        away={"PA":230,"AB":210,"H":47,"2B":10,"3B":2,"HR":11,"BB":18,"HBP":1,"SF":1,"SH":0,"K":64,"SB":2},
        gb_pct=0.34,fb_pct=0.45,ld_pct=0.21,platoon_boost=0.10,
        locked=True,verified_splits=True,notes="Career year 2025. Some regression possible.")

    tovar = _make("Ezequiel Tovar","SS",["SS","2B","3B"],"R",120,
        home={"PA":248,"AB":218,"H":71,"2B":15,"3B":1,"HR":13,"BB":22,"HBP":1,"SF":1,"SH":0,"K":53,"SB":10},
        away={"PA":196,"AB":179,"H":37,"2B":8,"3B":0,"HR":4,"BB":13,"HBP":1,"SF":0,"SH":0,"K":49,"SB":5},
        gb_pct=0.40,fb_pct=0.36,ld_pct=0.24,platoon_boost=0.07,
        locked=True,notes="⚠ Extreme Coors split. Injury risk. Home projecting to partially regress.")

    doyle = _make("Brenton Doyle","CF",["CF","LF","RF"],"R",135,
        home={"PA":265,"AB":237,"H":73,"2B":16,"3B":4,"HR":9,"BB":22,"HBP":3,"SF":2,"SH":0,"K":71,"SB":14},
        away={"PA":252,"AB":228,"H":38,"2B":8,"3B":1,"HR":6,"BB":16,"HBP":1,"SF":1,"SH":0,"K":84,"SB":8},
        gb_pct=0.38,fb_pct=0.36,ld_pct=0.26,platoon_boost=0.05,
        locked=True,notes="⚠ Away OPS still near replacement level. Gold Glove CF.")

    # ── Willi Castro (2yr/$12.8M, switch hitter)
    # Actual career platoon splits: .758 OPS vs RHP, .658 OPS vs LHP
    # Career-weighted blend: 2024 All-Star (40%), 2023 (30%), 2025 (30%)
    castro_away_vs_R = RawStats(PA=390,AB=340,H=87,B2=18,B3=3,HR=10,BB=44,HBP=4,SF=2,SH=0,K=78,SB=12)
    castro_away_vs_L = RawStats(PA=160,AB=144,H=36,B2=7, B3=1,HR=3, BB=14,HBP=2,SF=1,SH=0,K=30,SB=4)
    castro_home_vs_R = calibrate_home_from_away(castro_away_vs_R)
    castro_home_vs_L = calibrate_home_from_away(castro_away_vs_L)
    castro = Player(
        name="Willi Castro", pos="2B", eligible_pos=["2B","SS","LF","RF","CF","DH"],
        bats="S", games=130,
        home_vs_R=castro_home_vs_R, home_vs_L=castro_home_vs_L,
        away_vs_R=castro_away_vs_R, away_vs_L=castro_away_vs_L,
        gb_pct=0.43, fb_pct=0.34, ld_pct=0.23,
        speed_fps=get_speed("Willi Castro"),
        locked=True, verified_splits=False,
        notes="ACTUAL platoon splits: .758 OPS vs RHP, .658 OPS vs LHP (career). "
              "New to Coors (home calibrated). All-Star 2024.",
    )

    # ── Kyle Karros (3B, rookie) ───────────────────────────────────────────────
    # Calibration chain:
    #   AWAY projection:
    #     2025 AA Hartford (sea level, neutral): .294/.399/.462, ~225 PA
    #     Apply AA→MLB discount
    #     Blend with 2025 MLB road stats (small sample, 25% weight)
    #   HOME (Coors) projection:
    #     2025 AAA Albuquerque home stats (5,312 ft ≈ 5,280 ft Coors)
    #     Apply AAA quality discount (NOT altitude discount — already at elevation)
    #     Apply small ABQ→Coors geometry correction
    #
    # Scouting: Run grade 30 (well below avg), strong arm/glove. Struggles vs breaking balls.

    # AA Hartford 2025 neutral stats (55 games, ~225 PA)
    karros_aa_neutral = RawStats(PA=225, AB=193, H=57, B2=14, B3=2, HR=4, BB=32, HBP=3, SF=1, SH=0, K=45, SB=5)
    # AAA Albuquerque 2025 home games (altitude-matched; ~50 estimated home PA of his 16 games/62 AB)
    karros_abq_home = RawStats(PA=70, AB=62, H=19, B2=4, B3=1, HR=2, BB=7, HBP=1, SF=0, SH=0, K=14, SB=1)
    # 2025 MLB (42 games, ~150 PA)
    karros_mlb = RawStats(PA=150, AB=137, H=31, B2=6, B3=0, HR=1, BB=11, HBP=1, SF=1, SH=0, K=37, SB=2)

    # Away projection: AA discount → blend with MLB road (small sample weight)
    karros_aa_mlb  = apply_aa_to_mlb_discount(karros_aa_neutral)
    karros_away    = blend_seasons([karros_aa_mlb, karros_mlb], [0.75, 0.25])

    # Home projection: AAA ABQ discount (quality only) → ABQ→Coors geometry correction
    karros_abq_mlb_quality = apply_aaa_to_mlb_discount(karros_abq_home)
    karros_home_agg        = apply_abq_to_coors_correction(karros_abq_mlb_quality)
    # Blend ABQ-derived home with Coors-calibrated away as a sanity anchor (30% weight)
    karros_home_from_away  = calibrate_home_from_away(karros_away)
    karros_home            = blend_seasons([karros_home_agg, karros_home_from_away], [0.70, 0.30])

    def _rs_to_dict(s):
        return {"PA":s.PA,"AB":s.AB,"H":s.H,"2B":s.B2,"3B":s.B3,"HR":s.HR,
                "BB":s.BB,"HBP":s.HBP,"SF":s.SF,"SH":s.SH,"K":s.K,"SB":s.SB}

    karros = _make("Kyle Karros","3B",["3B","1B"],"R",42,
        home=_rs_to_dict(karros_home), away=_rs_to_dict(karros_away),
        gb_pct=0.42, fb_pct=0.36, ld_pct=0.22, platoon_boost=0.07,
        locked=True,
        notes="Calibration: AA Hartford (neutral) + ABQ home (altitude-matched). "
              "Run grade 30 (well below avg). Struggles vs breaking balls. "
              "⚠ High uncertainty — only 42 MLB games.")

    # ── TJ Rumfield (1B, spring winner) ───────────────────────────────────────
    # Calibration chain:
    #   All stats from sea-level parks (Scranton/WB, Somerset)
    #   Step 1: AAA → MLB quality discount
    #   Step 2: Away → Coors home calibration
    #
    # Data:
    #   2025 Triple-A SWB: .285/.378/.447 (approximate, ~400 PA)
    #   2024 Triple-A SWB: .292/.365/.461, 114 games, 421 AB
    #   Blend 2024 (55%) + 2025 (45%) before discount
    #
    # Scouting: Well below-avg speed, solid contact, fringy power, chases too much.
    # Speed: ~24.8 ft/sec (estimated; BA says "well below-average")

    rumfield_2024_aaa = RawStats(PA=464, AB=421, H=123, B2=26, B3=2, HR=15, BB=43, HBP=5, SF=3, SH=0, K=107, SB=3)
    rumfield_2025_aaa = RawStats(PA=400, AB=355, H=101, B2=22, B3=1, HR=12, BB=42, HBP=4, SF=2, SH=0, K=101, SB=2)
    rumfield_blend_aaa = blend_seasons([rumfield_2024_aaa, rumfield_2025_aaa], [0.55, 0.45])
    rumfield_away      = apply_aaa_to_mlb_discount(rumfield_blend_aaa)
    rumfield_home      = calibrate_home_from_away(rumfield_away)

    rumfield = _make("TJ Rumfield","1B",["1B","DH"],"L",0,
        home=_rs_to_dict(rumfield_home), away=_rs_to_dict(rumfield_away),
        gb_pct=0.38, fb_pct=0.42, ld_pct=0.20, platoon_boost=0.10,
        locked=True,
        notes="⚠ No MLB games — projected entirely from AAA. Sea-level → full Coors calibration. "
              "Well below-avg speed. Good contact, solid power for 1B.")

    # ── DH competitors (one will start each game) ─────────────────────────────

    freeman = _make("Tyler Freeman","DH",["DH","2B","3B","SS","LF","RF"],"R",115,
        home={"PA":220,"AB":198,"H":63,"2B":12,"3B":1,"HR":1,"BB":18,"HBP":2,"SF":1,"SH":0,"K":24,"SB":11},
        away={"PA":200,"AB":186,"H":47,"2B":9,"3B":1,"HR":1,"BB":12,"HBP":1,"SF":1,"SH":0,"K":22,"SB":6},
        gb_pct=0.52,fb_pct=0.26,ld_pct=0.22,platoon_boost=0.05,
        notes="Elite contact. Zero power. Classic leadoff/table-setter.")

    julien_blend = blend_seasons(
        [RawStats(PA=412,AB=356,H=94,B2=18,B3=2,HR=16,BB=54,HBP=5,SF=4,SH=0,K=98,SB=6),
         RawStats(PA=293,AB=251,H=50,B2=10,B3=1,HR=8, BB=38,HBP=3,SF=2,SH=0,K=71,SB=3),
         RawStats(PA=183,AB=159,H=35,B2=7, B3=0,HR=3, BB=21,HBP=2,SF=1,SH=0,K=44,SB=2)],
        [0.40, 0.35, 0.25],
    )
    julien = _make_from_away("Edouard Julien","DH",["DH","2B","1B"],"L",130,
        away_dict=_rs_to_dict(julien_blend),
        gb_pct=0.39,fb_pct=0.38,ld_pct=0.23,platoon_boost=0.12,
        notes="New to Coors (home calibrated). Elite walk rate. Bounce-back candidate.")

    mccarthy_blend = blend_seasons(
        [RawStats(PA=501,AB=442,H=126,B2=13,B3=7,HR=8,BB=45,HBP=3,SF=4,SH=0,K=118,SB=25),
         RawStats(PA=380,AB=337,H=84, B2=10,B3=4,HR=5,BB=37,HBP=2,SF=3,SH=0,K=94, SB=20),
         RawStats(PA=222,AB=196,H=40, B2=7, B3=4,HR=4,BB=19,HBP=2,SF=2,SH=0,K=60, SB=6)],
        [0.45, 0.30, 0.25],
    )
    mccarthy = _make_from_away("Jake McCarthy","DH",["DH","CF","LF","RF"],"L",130,
        away_dict=_rs_to_dict(mccarthy_blend),
        gb_pct=0.40,fb_pct=0.36,ld_pct=0.24,platoon_boost=0.14,
        notes="29.9 ft/sec (fastest on roster). Elite speed. "
              "3B rate at Coors exceptional (park factor 1.55x).")

    # Ritter stays as a backup/defensive-replacement option
    ritter = _make_from_away("Ryan Ritter","SS/2B",["SS","2B","3B","DH"],"R",60,
        away_dict={"PA":200,"AB":183,"H":44,"2B":12,"3B":1,"HR":2,
                   "BB":14,"HBP":2,"SF":1,"SH":0,"K":45,"SB":7},
        gb_pct=0.41,fb_pct=0.37,ld_pct=0.22,platoon_boost=0.06,
        notes="⚠ Only 60 MLB games. Backup/competing. Elite speed (29.2 ft/sec).")

    return [
        # Locked 8 (always in lineup)
        goodman, beck, moniak, tovar, doyle, castro, karros, rumfield,
        # DH competitors (optimizer picks one; others are bench)
        freeman, mccarthy, julien,
        # Deep bench / backup
        ritter,
    ]
