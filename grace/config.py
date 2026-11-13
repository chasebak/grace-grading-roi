"""Engine configuration: grading tiers, fees, priors. Everything is overridable.

All dollar amounts are USD; times are months; the discount rate is annual.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Tier:
    """One grading service level."""

    name: str
    fee: float  # grading fee per card
    grading_months: float  # typical turnaround


# Service levels and fees from PSA's public published fee schedule
# (https://www.psacard.com/pricing), with rough public turnaround
# estimates converted to months. All of this is configuration, not data.
# These batch-engine tiers are an earlier, illustrative reading, kept fixed
# so the synthetic study's numbers reproduce; the dated 2026-11-12 PSA table
# lives in the Grading Play Calculator (grace/play.py).
DEFAULT_TIERS = (
    Tier("regular", 79.99, 1.2),
    Tier("express", 149.0, 0.7),
    Tier("super_express", 299.0, 0.35),
    Tier("walk_through", 599.0, 0.25),
)

# Illustrative era base rates for the gem-rate prior, sensitivity-analyzed
# (see the m-estimation experiment). These are rounded, illustrative values
# with no numeric source attribution.
DEFAULT_ERA_PRIORS = {"modern": 0.45, "mid": 0.30, "vintage": 0.18}

# Era-dependent raw-price floors for the rankability gate.
DEFAULT_ERA_RAW_FLOORS = {"modern": 1.0, "mid": 2.0, "vintage": 2.0}


def _default_era_priors():
    return dict(DEFAULT_ERA_PRIORS)


def _default_era_raw_floors():
    return dict(DEFAULT_ERA_RAW_FLOORS)


@dataclass(frozen=True)
class EngineConfig:
    """All engine knobs with their shipped defaults."""

    tiers: tuple = DEFAULT_TIERS
    sale_months: float = 1.5  # time to sell after the grade comes back
    discount_rate_annual: float = 0.12
    sell_fee: float = 0.13  # marketplace fee fraction on the sale
    ship_pack: float = 5.0  # shipping and packing per sale
    other_fixed: float = 0.0  # extra fixed cost per submission
    m: float = 40.0  # pooling pseudo-count (fixed, not estimated)
    era_priors: dict = field(default_factory=_default_era_priors)
    lam: float = 0.0  # optional mean-risk penalty weight (lambda)
    a: float = 0.5  # capital exponent in the GRACE score
    roi_display_cap: float = 10.0  # annualized ROI is display-only, capped
    min_npv: float = 25.0  # rankability: minimum NPV
    # Rankability liquidity: minimum grade-10 sales in the input's
    # grade10_sales_180d column. The 180-day window is a property of that
    # input column, not an engine knob.
    min_liquidity: int = 3
    era_raw_floors: dict = field(default_factory=_default_era_raw_floors)


DEFAULT_CONFIG = EngineConfig()
