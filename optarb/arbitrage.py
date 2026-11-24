"""No-arbitrage bound checks on an option chain, priced at executable quotes.

Every check buys at the ask and sells at the bid, so an edge it reports is one you could
have traded at the snapshot, before commissions. Commissions, per-share stock costs and
stock borrow are then subtracted. Edges are in dollars per package (one "unit" of the
structure, with the standard 100-share multiplier).

Model-free checks (hold for any model, given a non-negative interest rate):

* ``strike_monotonicity``: call prices fall and put prices rise with strike.
* ``vertical_width``: a vertical spread is worth no more than the strike width.
* ``butterfly_convexity``: prices are convex in strike (any strike spacing).
* ``box_bounds``: a European box can't cost less than zero or pay more than its width.
* ``parity_band``: American put-call parity band, with stock, dividends and borrow.
* ``calendar``: an American option is worth at least the same option with a nearer expiry.

Rate-relative check (needs a funding rate you supply):

* ``box_vs_funding``: a European box lends or borrows at a rate far from your reference rate.

Exercise style matters. Boxes are checked only on European options, because early
assignment on American legs breaks the payoff guarantee. Calendars are checked only on
American options, because a deep in-the-money European put can rightly be worth less
with more time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from itertools import combinations

from optarb.models import CONTRACT_MULTIPLIER, ExerciseStyle, OptionQuote, OptionsChain, Right

BUY, SELL = "BUY", "SELL"


class Kind(str, Enum):
    MODEL_FREE = "model_free"
    RATE_RELATIVE = "rate_relative"


@dataclass(frozen=True)
class ScanConfig:
    commission_per_contract: float = 0.65
    commission_per_share: float = 0.0
    borrow_rate: float = 0.0  # annualized fee to borrow stock for a short stock leg
    min_net_edge: float = 0.01  # dollars per package after costs
    min_open_interest: int = 0
    min_volume: int = 0
    max_spread_pct: float = 100.0
    reference_rate: float | None = None  # funding rate for box_vs_funding; None turns it off
    box_rate_band: float = 0.005  # flag boxes whose net rate beats the reference by more than this


@dataclass(frozen=True)
class Leg:
    action: str  # BUY or SELL
    instrument: str  # CALL, PUT or STOCK
    quantity: int  # contracts, or shares for STOCK
    price: float  # per share, the executable side
    strike: float | None = None
    expiration: date | None = None

    @property
    def cash(self) -> float:
        """Signed cash flow in dollars: negative when you pay."""
        multiplier = 1 if self.instrument == "STOCK" else CONTRACT_MULTIPLIER
        sign = -1.0 if self.action == BUY else 1.0
        return sign * self.price * self.quantity * multiplier


@dataclass(frozen=True)
class Opportunity:
    check: str
    kind: Kind
    symbol: str
    asof: datetime
    expiration: date
    legs: tuple[Leg, ...]
    gross_edge: float  # dollars per package before commissions
    costs: float  # commissions, stock costs and borrow, dollars per package
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def net_edge(self) -> float:
        return self.gross_edge - self.costs

    @property
    def key(self) -> str:
        """Stable identity across snapshots, used to measure how long a violation lasts."""
        parts = [self.check, self.symbol, self.expiration.isoformat()]
        parts += [f"{leg.action[0]}{leg.instrument[0]}{leg.strike}@{leg.expiration}" for leg in self.legs]
        return "|".join(parts)


# ---------------------------------------------------------------------------
# helpers


def _usable(q: OptionQuote, action: str, cfg: ScanConfig) -> bool:
    price = q.ask if action == BUY else q.bid
    if price <= 0:
        return False
    if q.open_interest < cfg.min_open_interest or q.volume < cfg.min_volume:
        return False
    return q.spread_pct <= cfg.max_spread_pct


def _leg(q: OptionQuote, action: str, quantity: int = 1) -> Leg:
    return Leg(
        action=action,
        instrument="CALL" if q.right is Right.CALL else "PUT",
        quantity=quantity,
        price=q.ask if action == BUY else q.bid,
        strike=q.strike,
        expiration=q.expiration,
    )


def _costs(legs: tuple[Leg, ...], cfg: ScanConfig) -> float:
    total = 0.0
    for leg in legs:
        if leg.instrument == "STOCK":
            total += cfg.commission_per_share * leg.quantity
        else:
            total += cfg.commission_per_contract * leg.quantity
    return total


def _make(
    chain: OptionsChain,
    check: str,
    expiration: date,
    legs: tuple[Leg, ...],
    gross_edge: float,
    cfg: ScanConfig,
    kind: Kind = Kind.MODEL_FREE,
    extra_costs: float = 0.0,
    notes: tuple[str, ...] = (),
) -> Opportunity | None:
    opp = Opportunity(
        check=check,
        kind=kind,
        symbol=chain.symbol,
        asof=chain.asof,
        expiration=expiration,
        legs=legs,
        gross_edge=gross_edge,
        costs=_costs(legs, cfg) + extra_costs,
        notes=notes,
    )
    return opp if opp.net_edge >= cfg.min_net_edge else None


def _cents(x: float) -> int:
    return int(round(x * 100))


# ---------------------------------------------------------------------------
# model-free checks


def strike_monotonicity(chain: OptionsChain, cfg: ScanConfig) -> list[Opportunity]:
    """C(K1) >= C(K2) and P(K2) >= P(K1) for K1 < K2.

    If the higher-strike call bids above the lower-strike call's ask, buy the low call, sell the
    high call: a credit now and a call spread that can't pay less than zero.
    """
    out = []
    for exp in chain.expirations:
        for right in (Right.CALL, Right.PUT):
            side = chain.side(exp, right)
            for lo, hi in combinations(side, 2):
                cheap, rich = (lo, hi) if right is Right.CALL else (hi, lo)  # the one that must be worth more
                if not (_usable(cheap, BUY, cfg) and _usable(rich, SELL, cfg)):
                    continue
                credit = rich.bid - cheap.ask
                if credit <= 0:
                    continue
                legs = (_leg(cheap, BUY), _leg(rich, SELL))
                opp = _make(chain, "strike_monotonicity", exp, legs, credit * CONTRACT_MULTIPLIER, cfg)
                if opp:
                    out.append(opp)
    return out


def vertical_width(chain: OptionsChain, cfg: ScanConfig) -> list[Opportunity]:
    """C(K1) - C(K2) <= K2 - K1 and P(K2) - P(K1) <= K2 - K1.

    Selling the spread at the bid and buying it back at the ask must not collect more than the
    most it can ever cost you, the strike width. The undiscounted width is used so the bound also
    holds for American options.
    """
    out = []
    for exp in chain.expirations:
        for right in (Right.CALL, Right.PUT):
            side = chain.side(exp, right)
            for lo, hi in combinations(side, 2):
                width = hi.strike - lo.strike
                short, long_ = (lo, hi) if right is Right.CALL else (hi, lo)
                if not (_usable(short, SELL, cfg) and _usable(long_, BUY, cfg)):
                    continue
                credit = short.bid - long_.ask
                edge = credit - width
                if edge <= 0:
                    continue
                legs = (_leg(short, SELL), _leg(long_, BUY))
                opp = _make(chain, "vertical_width", exp, legs, edge * CONTRACT_MULTIPLIER, cfg)
                if opp:
                    out.append(opp)
    return out


def butterfly_convexity(chain: OptionsChain, cfg: ScanConfig) -> list[Opportunity]:
    """(K3 - K1) * V(K2) <= (K3 - K2) * V(K1) + (K2 - K1) * V(K3) on adjacent strikes K1 < K2 < K3.

    Works for uneven strike spacing: the wing quantities are the strike gaps, reduced to the
    smallest whole numbers of contracts. Holds for calls and puts, European or American.
    """
    out = []
    for exp in chain.expirations:
        for right in (Right.CALL, Right.PUT):
            side = chain.side(exp, right)
            for w1, body, w3 in zip(side, side[1:], side[2:], strict=False):
                if not (_usable(w1, BUY, cfg) and _usable(body, SELL, cfg) and _usable(w3, BUY, cfg)):
                    continue
                g12, g23 = _cents(body.strike - w1.strike), _cents(w3.strike - body.strike)
                if g12 <= 0 or g23 <= 0:
                    continue
                g = math.gcd(g12, g23)
                q1, q3 = g23 // g, g12 // g
                q2 = q1 + q3
                credit = q2 * body.bid - q1 * w1.ask - q3 * w3.ask
                if credit <= 0:
                    continue
                legs = (_leg(w1, BUY, q1), _leg(body, SELL, q2), _leg(w3, BUY, q3))
                opp = _make(chain, "butterfly_convexity", exp, legs, credit * CONTRACT_MULTIPLIER, cfg)
                if opp:
                    out.append(opp)
    return out


def _box_legs(c1, c2, p1, p2, long_box: bool) -> tuple[Leg, ...]:
    if long_box:
        return (_leg(c1, BUY), _leg(c2, SELL), _leg(p2, BUY), _leg(p1, SELL))
    return (_leg(c1, SELL), _leg(c2, BUY), _leg(p2, SELL), _leg(p1, BUY))


def _boxes(chain: OptionsChain, exp: date, cfg: ScanConfig):
    """Yield (width, long_cost, short_credit, legs_long, legs_short) for every usable strike pair."""
    pairs = chain.pairs(exp)
    for (c1, p1), (c2, p2) in combinations(pairs, 2):
        width = c2.strike - c1.strike
        long_ok = _usable(c1, BUY, cfg) and _usable(c2, SELL, cfg) and _usable(p2, BUY, cfg) and _usable(p1, SELL, cfg)
        short_ok = _usable(c1, SELL, cfg) and _usable(c2, BUY, cfg) and _usable(p2, SELL, cfg) and _usable(p1, BUY, cfg)
        long_cost = (c1.ask - c2.bid + p2.ask - p1.bid) if long_ok else None
        short_credit = (c1.bid - c2.ask + p2.bid - p1.ask) if short_ok else None
        yield width, long_cost, short_credit, _box_legs(c1, c2, p1, p2, True), _box_legs(c1, c2, p1, p2, False)


def box_bounds(chain: OptionsChain, cfg: ScanConfig) -> list[Opportunity]:
    """A European box pays exactly its width at expiry, so 0 <= price <= width when rates are non-negative.

    Long box (buy C1, sell C2, buy P2, sell P1) bought for a credit, or short box sold for more than its
    width, is a model-free gain. The short-box side is the one the original scanner never detected.
    """
    if chain.exercise_style is not ExerciseStyle.EUROPEAN:
        return []
    out = []
    for exp in chain.expirations:
        for width, long_cost, short_credit, legs_long, legs_short in _boxes(chain, exp, cfg):
            if long_cost is not None and long_cost < 0:
                opp = _make(
                    chain,
                    "box_bounds",
                    exp,
                    legs_long,
                    -long_cost * CONTRACT_MULTIPLIER,
                    cfg,
                    notes=(f"long box bought for a credit, width {width:g}",),
                )
                if opp:
                    out.append(opp)
            if short_credit is not None and short_credit > width:
                opp = _make(
                    chain,
                    "box_bounds",
                    exp,
                    legs_short,
                    (short_credit - width) * CONTRACT_MULTIPLIER,
                    cfg,
                    notes=(f"short box sold above its width {width:g}",),
                )
                if opp:
                    out.append(opp)
    return out


def parity_band(chain: OptionsChain, cfg: ScanConfig) -> list[Opportunity]:
    """American put-call parity band, traded with the stock.

    ``S - PV(div) - K <= C - P <= S - K * DF`` (Hull, chapter 11). Both sides are checked as
    conservatively as early exercise allows:

    * Conversion: buy stock at the ask, buy the put, sell the call. It locks in receiving K no
      later than expiry, so it is a gain when it costs less than ``K * DF``.
    * Reversal: short stock at the bid, sell the put, buy the call. You may have to pay K at any
      time (early assignment on the put) and you owe the dividends, so it is a gain only when the
      credit beats ``K`` plus the present value of dividends plus borrow.

    The discount factor comes from ``cfg.reference_rate``; without one, conversions are skipped
    because the bound depends on the rate. Dividends use the underlying's dividend yield.
    """
    if chain.exercise_style is not ExerciseStyle.AMERICAN:
        return []
    u = chain.underlying
    if u.bid <= 0 or u.ask <= 0:
        return []
    out = []
    for exp in chain.expirations:
        T = chain.year_fraction(exp)
        if T <= 0:
            continue
        pv_div = u.mid * (1.0 - math.exp(-u.dividend_yield * T))
        borrow = u.bid * cfg.borrow_rate * T
        for call, put in chain.pairs(exp):
            K = call.strike
            if cfg.reference_rate is not None and _usable(put, BUY, cfg) and _usable(call, SELL, cfg):
                cost = u.ask + put.ask - call.bid
                edge = K * math.exp(-cfg.reference_rate * T) - cost
                if edge > 0:
                    legs = (
                        Leg(BUY, "STOCK", CONTRACT_MULTIPLIER, u.ask),
                        _leg(put, BUY),
                        _leg(call, SELL),
                    )
                    opp = _make(
                        chain,
                        "parity_conversion",
                        exp,
                        legs,
                        edge * CONTRACT_MULTIPLIER,
                        cfg,
                        notes=(f"costs {cost:.2f} to receive {K:g} by expiry",),
                    )
                    if opp:
                        out.append(opp)
            if _usable(put, SELL, cfg) and _usable(call, BUY, cfg):
                credit = u.bid + put.bid - call.ask
                edge = credit - K - pv_div
                if edge > 0:
                    legs = (
                        Leg(SELL, "STOCK", CONTRACT_MULTIPLIER, u.bid),
                        _leg(put, SELL),
                        _leg(call, BUY),
                    )
                    opp = _make(
                        chain,
                        "parity_reversal",
                        exp,
                        legs,
                        edge * CONTRACT_MULTIPLIER,
                        cfg,
                        extra_costs=borrow * CONTRACT_MULTIPLIER,
                        notes=(f"credit {credit:.2f} against {K:g} plus dividends {pv_div:.2f}",),
                    )
                    if opp:
                        out.append(opp)
    return out


def calendar(chain: OptionsChain, cfg: ScanConfig) -> list[Opportunity]:
    """American options: V(K, T2) >= V(K, T1) for T2 > T1. Sell the near at the bid, buy the far at the ask."""
    if chain.exercise_style is not ExerciseStyle.AMERICAN:
        return []
    out = []
    exps = chain.expirations
    for right in (Right.CALL, Right.PUT):
        by_strike: dict[float, dict[date, OptionQuote]] = {}
        for exp in exps:
            for q in chain.side(exp, right):
                by_strike.setdefault(q.strike, {})[exp] = q
        for quotes in by_strike.values():
            for near_exp, far_exp in combinations(sorted(quotes), 2):
                near, far = quotes[near_exp], quotes[far_exp]
                if not (_usable(near, SELL, cfg) and _usable(far, BUY, cfg)):
                    continue
                credit = near.bid - far.ask
                if credit <= 0:
                    continue
                legs = (_leg(near, SELL), _leg(far, BUY))
                opp = _make(chain, "calendar", near_exp, legs, credit * CONTRACT_MULTIPLIER, cfg)
                if opp:
                    out.append(opp)
    return out


# ---------------------------------------------------------------------------
# rate-relative


@dataclass(frozen=True)
class BoxRate:
    expiration: date
    T: float
    best_lend_rate: float | None  # rate you earn buying the cheapest long box, after commissions
    best_borrow_rate: float | None  # rate you pay selling the richest short box, after commissions
    width: float | None


def box_rates(chain: OptionsChain, cfg: ScanConfig) -> list[BoxRate]:
    """Best executable lending and borrowing rates implied by European boxes, per expiry.

    Rates are continuously compounded and net of commissions on all four legs.
    """
    if chain.exercise_style is not ExerciseStyle.EUROPEAN:
        return []
    commission = 4 * cfg.commission_per_contract / CONTRACT_MULTIPLIER  # per share of box
    out = []
    for exp in chain.expirations:
        T = chain.year_fraction(exp)
        if T <= 0:
            continue
        lend, borrow, best_width = None, None, None
        for width, long_cost, short_credit, _, _ in _boxes(chain, exp, cfg):
            if long_cost is not None and 0 < long_cost + commission < width:
                rate = -math.log((long_cost + commission) / width) / T
                if lend is None or rate > lend:
                    lend, best_width = rate, width
            if short_credit is not None and 0 < short_credit - commission < width:
                rate = -math.log((short_credit - commission) / width) / T
                if borrow is None or rate < borrow:
                    borrow = rate
        out.append(BoxRate(exp, T, lend, borrow, best_width))
    return out


def box_vs_funding(chain: OptionsChain, cfg: ScanConfig) -> list[Opportunity]:
    """Flag European boxes whose net lending rate beats, or borrowing rate undercuts, the reference rate."""
    if chain.exercise_style is not ExerciseStyle.EUROPEAN or cfg.reference_rate is None:
        return []
    ref, band = cfg.reference_rate, cfg.box_rate_band
    out = []
    for exp in chain.expirations:
        T = chain.year_fraction(exp)
        if T <= 0:
            continue
        for width, long_cost, short_credit, legs_long, legs_short in _boxes(chain, exp, cfg):
            fair_long = width * math.exp(-ref * T)
            if long_cost is not None and long_cost > 0:
                rate = -math.log(long_cost / width) / T
                if rate > ref + band:
                    opp = _make(
                        chain,
                        "box_vs_funding",
                        exp,
                        legs_long,
                        (fair_long - long_cost) * CONTRACT_MULTIPLIER,
                        cfg,
                        kind=Kind.RATE_RELATIVE,
                        notes=(f"lends at {rate:.2%} vs reference {ref:.2%}",),
                    )
                    if opp:
                        out.append(opp)
            if short_credit is not None and 0 < short_credit < width:
                rate = -math.log(short_credit / width) / T
                if rate < ref - band:
                    opp = _make(
                        chain,
                        "box_vs_funding",
                        exp,
                        legs_short,
                        (short_credit - fair_long) * CONTRACT_MULTIPLIER,
                        cfg,
                        kind=Kind.RATE_RELATIVE,
                        notes=(f"borrows at {rate:.2%} vs reference {ref:.2%}",),
                    )
                    if opp:
                        out.append(opp)
    return out


MODEL_FREE_CHECKS = (strike_monotonicity, vertical_width, butterfly_convexity, box_bounds, parity_band, calendar)


def scan(chain: OptionsChain, cfg: ScanConfig | None = None) -> list[Opportunity]:
    """Run every applicable check and return opportunities sorted by net edge, largest first."""
    cfg = cfg or ScanConfig()
    found: list[Opportunity] = []
    for check in (*MODEL_FREE_CHECKS, box_vs_funding):
        found.extend(check(chain, cfg))
    return sorted(found, key=lambda o: o.net_edge, reverse=True)


def best_per_group(opportunities: list[Opportunity]) -> list[Opportunity]:
    """Keep the largest net edge for each check and expiry, so one mispriced quote shows once."""
    best: dict[tuple[str, date], Opportunity] = {}
    for o in opportunities:
        k = (o.check, o.expiration)
        if k not in best or o.net_edge > best[k].net_edge:
            best[k] = o
    return sorted(best.values(), key=lambda o: o.net_edge, reverse=True)
