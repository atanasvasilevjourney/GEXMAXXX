from dataclasses import dataclass


@dataclass
class Regime:
    state: str                     # "positive" | "negative"
    distance_to_zero_gamma: float  # signed points: + when spot above flip, - when below
    conviction: float              # 0.0–1.0; scales monotonically with |distance|
