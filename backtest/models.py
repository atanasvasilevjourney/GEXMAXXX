from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .validation import ValidationResult


@dataclass
class TradeRecord:
    entry_bar:       int
    exit_bar:        int
    direction:       str
    entry_price:     float
    exit_price:      float
    pnl_pts:         float
    regime_at_entry: str
    level_tier:      int
    level_strength:  float


@dataclass
class BacktestResult:
    total_trades:             int
    positive_regime_trades:   int
    negative_regime_trades:   int
    positive_regime_pnl:      float
    negative_regime_pnl:      float
    positive_regime_sharpe:   float
    negative_regime_sharpe:   float
    oos_positive_pnl:         float
    regime_split_significant: bool
    signal_only:              bool
    validation:               'ValidationResult | None' = field(default=None)
