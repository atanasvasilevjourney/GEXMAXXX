from .style import PricerStyle, detect_style
from .time import time_to_expiry, MIN_T_YEARS


def compute_gamma(*args, **kwargs):
    raise NotImplementedError


# compute_gamma and engines added in Task 2
__all__ = ['PricerStyle', 'detect_style', 'time_to_expiry', 'MIN_T_YEARS', 'compute_gamma']
