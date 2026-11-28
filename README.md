# optarb: option no-arbitrage checks at executable prices

[![CI](https://github.com/vincentdamato/OptionArbitrage/actions/workflows/ci.yml/badge.svg)](https://github.com/vincentdamato/OptionArbitrage/actions/workflows/ci.yml)

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

### Implied vol and Greeks from the quotes themselves

`optarb.greeks` prices every two-sided quote in a snapshot and returns its IV and Greeks next to the
ones the data source reported, when it reports them (Schwab does).

- European chains (SPX, XSP): Black-76 on the forward and discount factor implied by parity at that
  expiry, so no rate or dividend is assumed.
- American chains (equities, ETFs): IV by root-finding on the CRR tree; delta, gamma and theta from
  the tree's own early nodes, vega and rho by bumping. The rate is the one you pass, else the rate the
  source reported, else 4.5%, and each row says which (`rate_source`).
- `vendor_comparison` gives the median and 90th percentile gap and the median ratio for IV and each
  Greek. A ratio near 1 means the units agree; near 100 or 0.01 means a unit mismatch, not a pricing gap.
- `surface_summary` gives ATM IV, 25-delta put and call IV, the 25-delta risk reversal and butterfly
  per expiry, in vol points.

Real chains are messy, so the Schwab parser drops non-standard (adjusted) contracts and minis, turns
`-999` and `"NaN"` placeholders into missing values, and keeps quote sizes, quote times and the delayed
flag. The scan can then skip thin quotes (`--min-size`) and stale ones (`--max-quote-age` seconds).

## Quickstart

```bash
git clone https://github.com/vincentdamato/OptionArbitrage.git
cd OptionArbitrage
python -m venv .venv && source .venv/bin/activate
pip install -e ".[app,dev]"

pytest                       # 82 tests
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

Chains come from a swappable source: live Schwab market data, or Schwab JSON saved earlier and
replayed through the same parser. Everything downstream (snapshots, scan, Greeks, study) runs the same
code either way, which is how the whole workflow is tested without a Schwab login.

```bash
pip install -e ".[schwab]"
export SCHWAB_API_KEY=...  SCHWAB_APP_SECRET=...      # from your Schwab developer app
optarb doctor                                        # yes/no checks only; never prints a secret
optarb record SPY --login                            # first time only: browser sign-in, saves a token

# one trading day: record every 5 minutes until 4 pm New York, then write the study report
optarb session SPY QQQ '$XSP' --every 5 --until-close --save-raw --max-quote-age 120

optarb greeks data/recorded/SPY/SPY_20261120T153000Z.parquet --csv reports/spy_greeks.csv
optarb study data/recorded --out reports/recorded --rate 0.045
```

Replaying saved payloads instead of calling Schwab:

```bash
optarb session SPY --source recorded --replay data/recorded/raw/SPY --out data/replayed
optarb import-schwab saved_chain.json --out data/recorded    # one payload, one snapshot
```

The study report adds two sections when it has the data: the IV surface over the session (ATM, risk
reversal and butterfly per expiry) and how our IV and Greeks compare with Schwab's.

Schwab's refresh token lasts 7 days, after which `--login` is needed again; `optarb doctor` warns when
the token is close to that. The token lives in `~/.optarb/` (or `SCHWAB_TOKEN_PATH`), never in the repo.
`tests/fixtures/schwab_spy_chain.json` is a model-built chain laid out like a Schwab response, with the
quirks of real payloads; it is not market data.

Recorded chains stay on your machine. `data/` and `reports/` are in `.gitignore`, because exchange data
usually can't be redistributed. Share the write-up, not the quotes.

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
  greeks.py        IV and Greeks per quote, surface summary, comparison with vendor Greeks
  arbitrage.py     the checks, executable pricing, costs, box-implied rates
  smile.py         implied-vol smile per expiry (relative value, not arbitrage)
  volatility.py    close-to-close, Parkinson and Garman-Klass realized vol
  snapshot.py      Parquet snapshot I/O
  store.py         DuckDB over a folder of snapshots
  study.py         the research study and report
  cli.py           optarb command line
  providers/
    base.py        the chain-source interface record and session use
    schwab.py      read-only Schwab chain provider, parser and setup check
    recorded.py    replays saved Schwab JSON through the same parser
    synthetic.py   seeded synthetic chains with optional planted violations
app.py             Streamlit dashboard
tests/             pytest suite, run in CI on Python 3.11 and 3.12
  fixtures/        model-built Schwab-format chain and the script that makes it
```

## License

MIT. See [LICENSE](LICENSE).
