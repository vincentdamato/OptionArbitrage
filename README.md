# optarb: option no-arbitrage checks at executable prices

[![CI](https://github.com/vincentdamato/optionarbitrage/actions/workflows/ci.yml/badge.svg)](https://github.com/vincentdamato/optionarbitrage/actions/workflows/ci.yml)

`optarb` scans an option chain for violations of model-free no-arbitrage bounds, priced the way you
would actually trade them: buy at the ask, sell at the bid, then pay commissions and stock borrow. It
also asks the research question that a screener on its own can't answer: **how often do these bounds
really break in recorded data, by how much after costs, and for how long?**

It is a read-only research tool. It reads market data and never connects to an account or places an
order.

## What it checks

All prices are per share. Edges are reported in dollars per package, with the 100-share multiplier.

| Check | Bound | Applies to |
|---|---|---|
| Strike monotonicity | C(K1) >= C(K2) and P(K1) <= P(K2) for K1 < K2 | all |
| Vertical width | C(K1) - C(K2) <= K2 - K1, and the same for puts | all |
| Butterfly convexity | (K3 - K1) V(K2) <= (K3 - K2) V(K1) + (K2 - K1) V(K3) | all, any strike spacing |
| Box bounds | 0 <= box <= K2 - K1 | European only |
| Parity band | S - D - K <= C - P <= S - K e^(-rT) | American only |
| Calendar | V(K, T2) >= V(K, T1) for T2 > T1 | American only |
| Box vs funding | box-implied rate against your reference rate | European, rate-relative |

Exercise style decides which bounds hold. Boxes are only checked on European options (SPX, XSP),
because early assignment on American legs breaks the guaranteed payoff. Calendars are only checked
on American options, because a deep in-the-money European put can rightly be worth less with more
time to expiry. The butterfly check uses the strike gaps as weights, so uneven strikes are handled
without false positives.

The model-free checks need no rate, volatility or dividend input. Rate-relative checks are kept
separate and only run when you supply a reference rate.

### Market-implied forwards

For each expiry, the implied forward F and discount factor DF come from a weighted least-squares fit
of put-call parity, `C - P = DF * F - DF * K`, across strikes near the money, weighted by the inverse
bid/ask spread. Strikes far off the line (a stale or mispriced quote) are dropped before the final fit.
This replaces a hard-coded risk-free rate and a guessed dividend yield with what the market implies.
For American options the fit is labeled approximate.

### Pricing and Greeks

`optarb.pricing` has Black-Scholes-Merton with a continuous dividend yield, Black-76, a vectorized
Cox-Ross-Rubinstein tree for American options and closed-form Greeks (delta, gamma, theta, vega, rho,
vanna, charm, vomma, speed). Units are explicit: theta and charm per calendar day, vega and vanna per
volatility point, rho per percentage point. Every analytic Greek is tested against finite differences.

## Quickstart

```bash
git clone https://github.com/vincentdamato/optionarbitrage.git
cd optionarbitrage
python -m venv .venv && source .venv/bin/activate
pip install -e ".[app,dev]"

pytest                       # 56 tests
streamlit run app.py         # dashboard on the demo snapshot

optarb sample --out data/sample
optarb scan data/sample/XSP/*.parquet
optarb forwards data/sample/XSP/*.parquet
```

The demo snapshot is **synthetic**: a model-priced, XSP-like chain with three violations planted on
purpose so the checks have something to find. The app and the CLI label it as synthetic everywhere it
appears. A clean synthetic chain (no planted violations) produces zero model-free flags, and that is
one of the tests.

## The research study

```bash
optarb simulate --out data/simulated --n 40          # synthetic series, to see the pipeline work
optarb study data/simulated --out reports/simulated --rate 0.045
```

`optarb study` reads a folder of snapshots with DuckDB, runs every check on each one and writes
`report.md` plus CSVs:

- **Frequency:** the share of snapshots with at least one violation, by check.
- **Size:** the best net edge after costs, in dollars per package.
- **Persistence:** episodes of the same check broken at the same expiry in consecutive snapshots, and
  how long they lasted.
- **Box-implied rates:** the best lending and borrowing rates boxes offered after commissions, per
  expiry, against your reference rate.

The simulated series has known, planted violations, so the persistence measure has a known answer to
recover (it is tested). **The real result needs recorded data.**

### Recording real chains

`optarb record` saves option chains from the Schwab Trader API to Parquet. It only reads market data.

```bash
pip install -e ".[schwab]"
export SCHWAB_API_KEY=...  SCHWAB_APP_SECRET=...      # from your Schwab developer app
optarb record SPY QQQ '$XSP' --every 5 --count 78 --out data/recorded --login
optarb study data/recorded --out reports/recorded --rate 0.045
```

Recorded chains stay on your machine. `data/` is in `.gitignore`, because exchange data usually can't
be redistributed. Commit the report, not the quotes.

## What a flag does not tell you

- Quotes in a snapshot are not guaranteed to be simultaneous, and the bid or ask may be for one contract.
- A flag says the quotes crossed a bound, not that you would have been filled.
- American parity bounds depend on the dividend you supply; a missed dividend shows up as a false
  reversal.
- Short stock legs assume the stock is borrowable at the fee you set.

These are exactly what the study is meant to measure, which is why it reports persistence and size
after costs rather than a count of flags.

## Layout

```
optarb/
  models.py        quotes, chains, exercise style, year fractions
  pricing.py       BSM, Black-76, CRR American, Greeks, implied vol
  forwards.py      implied forward and discount factor per expiry
  arbitrage.py     the checks, executable pricing, costs, box-implied rates
  smile.py         implied-vol smile per expiry (relative value, not arbitrage)
  volatility.py    close-to-close, Parkinson and Garman-Klass realized vol
  snapshot.py      Parquet snapshot I/O
  store.py         DuckDB over a folder of snapshots
  study.py         the research study and report
  cli.py           optarb command line
  providers/
    schwab.py      read-only Schwab chain provider and parser
    synthetic.py   seeded synthetic chains with optional planted violations
app.py             Streamlit dashboard
tests/             pytest suite, run in CI on Python 3.11 and 3.12
```

## License

MIT. See [LICENSE](LICENSE).
