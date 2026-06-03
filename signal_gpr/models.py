from enum import Enum
from dataclasses import dataclass


class SignalState(str, Enum):
    IDLE     = "IDLE"
    ARMED    = "ARMED"
    IN_TRADE = "IN_TRADE"
    STANDBY  = "STANDBY"


class Action(str, Enum):
    NONE        = "NONE"
    ENTER_LONG  = "ENTER_LONG"
    ENTER_SHORT = "ENTER_SHORT"
    EXIT        = "EXIT"
    HOLD        = "HOLD"


@dataclass
class MarketTick:
    price: float
    atr: float
    has_rejection_bar: bool = False
    in_event_window: bool = False


@dataclass
class TradePosition:
    direction: str
    entry_price: float
    stop: float
    target: float | None
    level_strength: float
