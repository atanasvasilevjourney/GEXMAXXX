from dataclasses import dataclass


@dataclass
class Level:
    strike: float
    gex: float
    tier: int
    strength: float
    label: str


@dataclass
class FutLevel:
    fut_price: float
    tier: int
    strength: float
    label: str
    source_strike: float
    basis_pts: float
