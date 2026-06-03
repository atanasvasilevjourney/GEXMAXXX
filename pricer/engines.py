from __future__ import annotations
import math
import QuantLib as ql
from .time import MIN_T_YEARS


def _ql_date_from_T(T: float) -> ql.Date:
    """Convert T in years to a QuantLib Date offset from today."""
    days = max(int(T * 365.25), 1)
    return ql.Date.todaysDate() + days


def _bsm_process(S: float, r: float, sigma: float) -> ql.BlackScholesMertonProcess:
    """Build a Black-Scholes-Merton process with zero dividend yield."""
    spot   = ql.QuoteHandle(ql.SimpleQuote(S))
    r_ts   = ql.YieldTermStructureHandle(
        ql.FlatForward(0, ql.NullCalendar(),
                       ql.QuoteHandle(ql.SimpleQuote(r)),
                       ql.Actual365Fixed()))
    div_ts = ql.YieldTermStructureHandle(
        ql.FlatForward(0, ql.NullCalendar(),
                       ql.QuoteHandle(ql.SimpleQuote(0.0)),
                       ql.Actual365Fixed()))
    vol_ts = ql.BlackVolTermStructureHandle(
        ql.BlackConstantVol(0, ql.NullCalendar(),
                            ql.QuoteHandle(ql.SimpleQuote(sigma)),
                            ql.Actual365Fixed()))
    return ql.BlackScholesMertonProcess(spot, div_ts, r_ts, vol_ts)


def ql_european_gamma(S: float, K: float, T: float, r: float,
                      sigma: float, option_type: str) -> float:
    """
    European option gamma via QuantLib AnalyticEuropeanEngine.

    Use for SPX, NDX — cash-settled index options with no early exercise.
    """
    if sigma <= 0 or S <= 0:
        return 0.0
    ql.Settings.instance().evaluationDate = ql.Date.todaysDate()
    opt_type = ql.Option.Call if option_type == 'call' else ql.Option.Put
    payoff   = ql.PlainVanillaPayoff(opt_type, K)
    exercise = ql.EuropeanExercise(_ql_date_from_T(T))
    process  = _bsm_process(S, r, sigma)
    option   = ql.VanillaOption(payoff, exercise)
    option.setPricingEngine(ql.AnalyticEuropeanEngine(process))
    return option.gamma()


def ql_american_gamma(S: float, K: float, T: float, r: float,
                      sigma: float, option_type: str) -> float:
    """
    American option gamma via Barone-Adesi-Whaley approximation.

    Use for QQQ, SPY, IWM — ETF options where early exercise is possible.
    Gamma computed via central-difference numerical differentiation of BAW price:
        gamma ≈ (P(S+ε) + P(S-ε) - 2·P(S)) / ε²  where ε = S × 0.001
    """
    if sigma <= 0 or S <= 0:
        return 0.0
    ql.Settings.instance().evaluationDate = ql.Date.todaysDate()
    opt_type = ql.Option.Call if option_type == 'call' else ql.Option.Put
    maturity = _ql_date_from_T(T)
    today    = ql.Date.todaysDate()

    def _price(s: float) -> float:
        process = _bsm_process(s, r, sigma)
        option  = ql.VanillaOption(
            ql.PlainVanillaPayoff(opt_type, K),
            ql.AmericanExercise(today, maturity),
        )
        option.setPricingEngine(ql.BaroneAdesiWhaleyApproximationEngine(process))
        return option.NPV()

    eps = S * 0.001
    return (_price(S + eps) + _price(S - eps) - 2.0 * _price(S)) / (eps ** 2)


def ql_black76_gamma(S: float, K: float, T: float, r: float,
                     sigma: float, option_type: str) -> float:
    """
    Black-76 gamma for futures options (closed-form).

    S is the futures price (forward). The discount factor e^{-rT} is included
    per the correct Black-76 formula:
        gamma = e^{-rT} * N'(d1) / (F * sigma * sqrt(T))
    where d1 = (ln(F/K) + 0.5 * sigma^2 * T) / (sigma * sqrt(T))
    """
    if sigma <= 0 or S <= 0:
        return 0.0
    d1      = (math.log(S / K) + 0.5 * sigma ** 2 * T) / (sigma * math.sqrt(T))
    npdf_d1 = math.exp(-0.5 * d1 ** 2) / math.sqrt(2 * math.pi)
    return math.exp(-r * T) * npdf_d1 / (S * sigma * math.sqrt(T))
