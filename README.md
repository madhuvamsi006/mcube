# mcube

**Short-horizon directional prediction for NSE index derivatives (NIFTY / BANKNIFTY / SENSEX), built to find out whether a tradeable edge survives Indian transaction costs — and to be honest when it doesn't.**

Research code. Not a trading system. Nothing here places orders.

---

## TL;DR — what we found

Trained on 3 years of 1-minute NIFTY data (277,087 bars, 741 sessions) from the Dhan API.

| | |
|---|---|
| **Edge found** | +9 to +12 percentage points over base rate |
| **AUC** | 0.5516 (real) vs 0.4989 (shuffled) — genuine, not leakage |
| **Works at** | 40–60 point targets, 30–60 minute horizons |
| **Does NOT work at** | 10–20 point targets, 15 minute horizons |

The original goal was *"predict direction, grab 10–20 points in 15 minutes."* The data says no — not because the model is weak, but because Indian transaction costs are ~5–7 index points per round trip, and a 20-point target needs +15pp of skill to clear them. The model delivers +9pp.

Widen the target to 40–60 points and the same model becomes profitable, because required skill falls faster than measured skill does.

Full numbers and caveats: **[RESULTS.md](RESULTS.md)**

---

## Table of contents

1. [The problem](#1-the-problem)
2. [How it works](#2-how-it-works)
3. [The model](#3-the-model)
4. [How we trained it](#4-how-we-trained-it)
5. [How we tested it](#5-how-we-tested-it-the-important-part)
6. [Results](#6-results)
7. [Why it fails at short targets](#7-why-it-fails-at-short-targets)
8. [Repo layout](#8-repo-layout)
9. [Setup and running](#9-setup-and-running)
10. [Plan of action](#10-plan-of-action)
11. [Known limitations](#11-known-limitations)

---

## 1. The problem

Predicting raw price direction is close to a coin flip. The tractable version is **conditional drift**:

> Given the current market state, is the probability of price hitting **+X points before −Y points** meaningfully above the level needed to cover costs?

Three things make this hard in Indian index derivatives specifically:

**Costs are large relative to the move.** A round trip on NIFTY futures costs ~7 index points; options ~5 points (delta-adjusted). If you're targeting 10 points, over half your gross is gone before you start.

**Costs hurt twice.** They shrink wins *and* enlarge losses. A nominal 1.5:1 trade pays much worse than 1.5:1 after charges.

**The base rate isn't 50%.** With an asymmetric target/stop, the nearer barrier is hit more often. For a 20/13 setup the gambler's-ruin base rate is 39.4% — so "55% breakeven" really means *+16 percentage points of genuine skill*.

### What "institutional" can and cannot mean here

**Not observable on NSE/BSE:** order-by-order (MBO) data. No retail feed exposes it, so true iceberg detection and queue modelling are impossible. Anything claiming "institutional order detection" from aggregated depth is *inferring*, not observing.

**Observable, and genuinely strong** — India exposes positioning data US desks don't get:

| Signal | Why it matters |
|---|---|
| Strike-level ΔOI | Institutions express views by *writing* options |
| Participant-wise OI | NSE publishes F&O OI split FII/DII/Pro/Client daily |
| Put-call parity gap | Synthetic futures (`K + C − P`) vs actual — shows aggressive one-sided flow |
| Futures basis | Institutional hedging pressure |

Code for all of these is in `features/institutional.py`. **They are not in the v1 result** — see [limitations](#11-known-limitations).

---

## 2. How it works

```
Dhan API  ──►  1-min OHLCV bars  ──►  Features  ──►  Triple-barrier labels
                                                            │
                                            Purged walk-forward CV
                                                            │
                                                      LightGBM
                                                            │
                                            Cost model  ──►  Verdict
```

### Labeling: triple-barrier

A fixed-horizon label (*"+1 if the 15-min return is positive"*) ignores path and doesn't describe a real trade. We use **triple-barrier labeling** (López de Prado): from each bar, place three barriers —

```
  upper  = entry + target      →  label +1  (profit)
  lower  = entry − stop        →  label −1  (loss)
  vertical = entry + N bars    →  label  0  (timeout)
```

Whichever is touched first becomes the label. **The label is literally the trade spec** — the model is trained on exactly what you'd execute. Change the target in config, and the labels change with it.

Implementation: `labeling/triple_barrier.py`

### Features (v1 — 37 total)

All strictly backward-looking. Any centred or forward-looking window would leak.

| Group | Examples | Intuition |
|---|---|---|
| **Bar shape** | `close_loc`, `body_ratio`, wicks | Where price closed in its range = who held control |
| **Returns** | `ret_{1,3,5,15,30,60}` + z-scores | Momentum / mean-reversion |
| **Volatility** | `rv_{15,60,240}`, `parkinson_15`, `rv_ratio` | Regime; which world are we in |
| **Volume** | `vol_z_30`, `vol_z_240`, `vol_ratio` | Participation |
| **Flow proxy** | `signed_vol_*`, `flow_imb_*` | Bar direction × volume — crude stand-in for signed order flow |
| **VWAP** | `vwap_dev_bps` | Distance from the institutional execution benchmark |
| **Mean reversion** | `zscore_{15,60}`, `streak` | Short-run streaks revert more than they continue |
| **Session** | `tod_sin/cos`, `is_midday`, `minutes_to_close` | Time of day is a top-5 feature in Indian index trading |
| **OI** (futures only) | `doi_*`, `buildup_score` | Long buildup vs short covering |

Implementation: `features/bars.py`

### Cost model

Modelled **per trade, inside the loop** — never subtracted from a final equity curve.

The key asymmetry: **futures STT is charged on notional (~₹19.5L/lot); options STT on premium (~₹11k/lot).** A ~35× difference on the dominant line item, which is why small captures are cheaper to express in options.

| Charge | Futures | Options |
|---|---|---|
| STT (sell) | 0.02% of notional | 0.1% of **premium** |
| Exchange txn | 0.00173% | 0.03503% of premium |
| Brokerage | ₹20/order | ₹20/order |
| Stamp, SEBI, GST 18% | yes | yes |
| Slippage | 1 tick | ~0.67% of premium |

Implementation: `costs/india_costs.py`. Rates reflect the post-Oct-2024 STT revision — **verify against your broker's live charge sheet.**

---

## 3. The model

**LightGBM (gradient-boosted trees), binary classification.**

```python
objective:         binary
n_estimators:      400
learning_rate:     0.03
num_leaves:        31
min_child_samples: 200     # high — microstructure features are very noisy
subsample:         0.8
colsample_bytree:  0.6
reg_lambda:        5.0
```

**Why not a Transformer / LSTM?** On tabular features with this signal-to-noise ratio and this data size, gradient boosting wins. It trains in minutes, and its feature importances tell you *when it breaks*. Deep sequence models are worth revisiting only once a tabular baseline shows real edge.

**Target:** `P(target hit before stop)`. Timeout (label 0) samples are dropped from the primary model — direction is undefined when neither barrier is touched.

### Meta-labeling (built, not yet in the v1 result)

Two-stage design: the **primary** model picks direction; a **meta** model decides *whether to take the trade at all*.

This is where ML earns its keep. Pushing direction accuracy past ~55% is very hard; learning to *skip* bad setups is tractable and does more for net P&L, because every avoided trade saves a full round trip of costs.

Implementation: `models/train.py`

---

## 4. How we trained it

**Data:** Dhan `/v2/charts/intraday`, NIFTY spot (securityId `13`, segment `IDX_I`), 1-minute bars, 2023-11-06 → 2026-11-03. 277,087 bars across 741 sessions. Auto-chunked past Dhan's 90-day-per-request cap, throttled to the 5 req/sec limit.

**Validation: walk-forward.** Train 12 months → test the next 3 → roll forward. 9 folds.

```
fold 1:  [train 2023-11 .. 2024-10] [test 2024-11 .. 2025-01]
fold 2:  [train 2024-02 .. 2025-01] [test 2025-02 .. 2025-04]
...
```

This never trains on data after the test period, so it reflects how the model would actually have been deployed. **K-fold is used only for hyperparameter choice, never for reported P&L.**

**Purging + embargo.** Triple-barrier labels span multiple bars: a label stamped 10:00 may not resolve until 10:15. A naive split puts the 10:00 sample in train and the 10:05 sample in test while their label windows overlap — the training set then contains the test set's future.

- **Purge:** drop training samples whose label window overlaps the test window
- **Embargo:** drop a further 30 bars after the test window (serial correlation bleeds past the boundary)

`assert_no_leakage()` runs inside every fold and raises if any overlap survives.

Implementation: `validation/purged_cv.py`

---

## 5. How we tested it (the important part)

Most retail trading bots fail the same way: the backtest shows huge profits, live trading loses money. The cause is almost always **leakage** — the model accidentally sees future information.

We ran **three independent integrity checks** before believing any result.

### Test 1 — Synthetic null

Generate fake market data with **zero real signal** (pure random walk, features independent of returns). Run the entire pipeline unchanged.

```
Expected: AUC ≈ 0.50   (finds nothing)
Actual:   AUC = 0.4989  ✅ PASS
```

If the pipeline "discovers a strategy" in noise, it's cheating.

### Test 2 — Synthetic signal

Same generator, but with a known weak latent signal embedded.

```
Expected: AUC > 0.50   (detects edge when edge exists)
Actual:   AUC = 0.637  ✅ PASS
```

Confirms the pipeline isn't merely blind.

### Test 3 — Real-data shuffle

Take the **real** NIFTY data. Permute only the label *values*, keeping timestamps and `t1` intact so the CV structure is identical. Only the feature↔label relationship is destroyed.

```
Expected: AUC ≈ 0.50
Actual:   AUC = 0.4989  ✅ PASS
Real run: AUC = 0.5516
```

> **Note:** our first shuffle test was wrong — it permuted the *index*, which broke the `t1` mapping that purging depends on, and reported a false leak (AUC 0.628). Shuffle label values only.

### Why we report AUC, not accuracy

```
accuracy on REAL labels:      0.6163
accuracy on SHUFFLED labels:  0.6186   ← higher!
```

Triple-barrier labels are imbalanced by construction (the nearer stop is hit more often), so a model that learns nothing but the base rate scores ~62%. **Accuracy is meaningless here.** AUC is base-rate invariant.

Run them yourself:

```bash
python scripts/run_pipeline.py --mode null             # must be ~0.50
python scripts/run_pipeline.py --mode signal           # should be >0.50
python scripts/train_real.py --shuffle-test            # must be ~0.50
```

---

## 6. Results

Top 5% confidence long signals, walk-forward. Costs: ATM options, 5 lots, 4.6 index points round trip.

| target/stop | horizon | base rate | hit rate | **edge** | breakeven | ₹/trade |
|---|---|---|---|---|---|---|
| 20 / 13 | 15 min | 38.1% | 47.3% | **+9.2** | 53.5% | −662 ❌ |
| 30 / 20 | 15 min | 34.9% | 43.7% | **+8.7** | 49.3% | −913 ❌ |
| 30 / 20 | 30 min | 38.0% | 43.9% | **+5.9** | 49.3% | −872 ❌ |
| 40 / 27 | 30 min | 36.2% | 48.2% | **+12.1** | 47.2% | **+220** ✅ |
| 50 / 33 | 45 min | 35.8% | 48.0% | **+12.2** | 45.4% | **+699** ✅ |
| 60 / 40 | 60 min | 36.3% | 47.0% | **+10.7** | 44.6% | **+773** ✅ |

**Top feature importances:** `rv_240`, `rv_60`, `ret_z_60`, `ret_60`, `parkinson_15`, `session_progress`, `rv_ratio`, `tod_cos`

Volatility regime and time-of-day dominate — the model is mostly learning *when* conditions favour follow-through, not predicting direction from price patterns.

**Two findings worth flagging:**

- **Top 1% confidence scored *worse* than top 5%** (44.2% vs 47.3%). The extreme tail overfits. Use the top-5% band.
- **One fold out of nine had zero edge** (38.5% — exactly base rate). The edge is not stable across regimes. Expect losing quarters.

---

## 7. Why it fails at short targets

Required skill = `breakeven hit rate − random-walk base rate`

```python
random_walk_base_rate = stop / (target + stop)          # gambler's ruin
breakeven             = (stop + cost) / ((target − cost) + (stop + cost))
```

| target/stop | base rate | breakeven | **skill needed** | skill we have |
|---|---|---|---|---|
| 10 / 7 | 41.2% | 71.8% | **+30.6pp** | +9pp ❌ |
| 20 / 13 | 39.4% | 53.5% | **+14.1pp** | +9pp ❌ |
| 40 / 27 | 40.3% | 47.2% | **+6.9pp** | +12pp ✅ |
| 60 / 40 | 40.0% | 44.6% | **+4.6pp** | +11pp ✅ |

The base rate stays pinned near 40% for any 1.5:1 ratio. But **cost shrinks as a share of the move**, so required skill falls as targets widen. That's the whole story.

**More lots does not help.** Lots scale P&L and costs together — the ratio never changes. (Flat ₹20/order brokerage amortises slightly: ~7% better from 1→5 lots. Real, but not decisive.)

---

## 8. Repo layout

```
src/mcube/
  costs/india_costs.py        STT/GST/slippage; breakeven, expectancy, required_edge
  labeling/triple_barrier.py  vol-scaled barriers; label == trade spec
  validation/purged_cv.py     purged K-fold, walk-forward, leakage assert
  features/
    bars.py                   v1: 37 OHLCV-derived features  ← used in results
    microstructure.py         OFI, depth imbalance, microprice (needs depth feed)
    institutional.py          strike ΔOI, PCR, parity gap, basis, participant OI
    regime.py                 time-of-day, days-to-expiry, vol regime, cross-index
  models/train.py             LightGBM primary + meta-labeling
  backtest/
    engine.py                 walk-forward, costs per trade, next-bar fills
    sweep.py                  target feasibility map (runs with no data)
  data/
    dhan.py                   DhanHQ v2 client (bars, option chain, scrip master)
    dhan_auth.py              TOTP token auto-refresh (tokens expire in 24h)
    synthetic.py              null/signal generator for pipeline validation
scripts/
  fetch_history.py            download Dhan history → parquet
  run_pipeline.py             end-to-end on synthetic data (null + signal tests)
  train_real.py               train on real data; --shuffle-test for leakage
tests/                        12 tests pinning cost math and leakage guards
```

---

## 9. Setup and running

```bash
pip install -e ".[dev]"
```

**Credentials** — copy `.env.example` to `.env` (gitignored):

```
DHAN_CLIENT_ID=100xxxxxxx
DHAN_ACCESS_TOKEN=eyJ...      # web.dhan.co → profile → Access DhanHQ APIs
```

Tokens expire in 24h. For unattended jobs use TOTP auto-refresh instead (`DHAN_PIN` + `DHAN_TOTP_SECRET`) — see `dhan_auth.py`.

> ⚠️ **Client ID + PIN + TOTP seed together grant full account access, including placing orders.** Never commit them. In GCP use Secret Manager, not a file on the VM. Prefer a data-only key if available.

```bash
# No data needed — cost feasibility map
python -m mcube.backtest.sweep --symbol NIFTY --annual

# Validate the pipeline on synthetic data
python scripts/run_pipeline.py --mode null       # must be ~0.50
python scripts/run_pipeline.py --mode signal

# Real data
python scripts/fetch_history.py --years 3
python scripts/train_real.py --symbol NIFTY --target 50 --stop 33
python scripts/train_real.py --shuffle-test      # leakage check

pytest
```

---

## 10. Plan of action

### Phase 0 — done ✅

- [x] Cost model, verified against live lot sizes from Dhan's scrip master
- [x] Triple-barrier labeling, purged CV, leakage guards
- [x] Synthetic null/signal harness
- [x] Dhan client with auto-chunking, throttling, token refresh
- [x] 3 years of NIFTY data, model trained, edge measured
- [x] 12 tests green

### Phase 1 — collector ✅ BUILT

**Highest priority, because the cost of waiting is permanent.** Dhan serves historical *bars*, but order book depth and intraday strike-level OI are **live-only**. Every day without a collector is training data lost forever.

- [x] Live feed websocket, binary packet parser (10 tests pinning the wire format)
- [x] Quote + 5-level depth + OI, auto-chunked subscription (100/message cap)
- [x] Dynamic option-chain universe: ATM ± N strikes, rolls past expired series
- [x] Buffer → local parquet → optional GCS (never network on the receive path)
- [x] Auto-reconnect with exponential backoff; token refresh on auth failure
- [x] Gap monitor — alerts when a connected-but-silent feed stalls
- [x] Verified live: 24 instruments subscribed, depth ladders correct
- [x] Cloud Run Job + Cloud Scheduler, `deploy/cloudrun.sh` (~$5-8/month)
- [ ] 200-level depth on the future (separate socket, 1 instrument/connection)

**Why a Cloud Run Job rather than a VM:** billed only for the ~6.5h it runs
instead of 24h, nothing to patch, and Cloud Run retries a task that dies.
Not a *Service* — a Service is request-driven and scales to zero when no
request is in flight, whereas this process holds an outbound websocket for
hours with no inbound traffic. Jobs allocate CPU for the whole task.

The container filesystem is in-memory tmpfs that vanishes with the task, so
`--cloud` ships each parquet to GCS a minute after writing and deletes the
local copy. At most one minute of data is ever unreplicated.

Credentials live in Secret Manager, bound to a dedicated service account with
only `storage.objectAdmin` on the bucket and `secretAccessor` on three
secrets — not the default compute SA.

```bash
export GCP_PROJECT_ID=your-project GCS_BUCKET=your-bucket
./deploy/cloudrun.sh
```

```bash
python scripts/run_collector.py --dry-run          # 60s connectivity check
python scripts/run_collector.py --symbols NIFTY    # full session
```

Writes to `data/live/dt=YYYY-MM-DD/`: `ticks_*.parquet` (quote + depth + OI),
`universe.parquet` (security_id → strike/expiry — ids are reused across
expiries, so this must be captured at record time), `chain_*.parquet`.

### Phase 2 — fill the data gaps (~2 days)

- [ ] Stitch expired futures contracts for multi-year OI and basis history
      *(Dhan's Expired Options Data API)*
- [ ] Daily FII/DII and participant-wise OI from NSE (free, published after close — **lag by one day**, it isn't available during the session it describes)
- [ ] Rebuild features with `institutional.py` wired in

### Phase 3 — honest backtest (~2 days)

- [ ] Full walk-forward at 50/33, 45-min horizon
- [ ] Daily trade caps, one position at a time, next-bar fills
- [ ] Realistic option slippage, verified against live quotes
- [ ] Expiry days evaluated as a **separate slice** (gamma regime differs; pooling flatters metrics)
- [ ] Enable meta-labeling; measure how much skipping bad setups adds

### Phase 4 — paper trading (1 month minimum)

- [ ] Live feed, simulated fills, zero capital
- [ ] Compare live signal distribution to backtest — divergence means feature skew
- [ ] Track slippage actually observed vs modelled

### Phase 5 — live, small (only if Phase 4 agrees)

- [ ] 1 lot. Not 3, not 5.
- [ ] Kill switch, daily loss limit, position reconciliation
- [ ] Scale only after a statistically meaningful sample

### Architecture when it goes live

**One VM, one process — not microservices:**

```
WebSocket → ring buffers → features → LightGBM in-process → risk → order
                              │
                              └── async → BigQuery (logging only, off hot path)
```

The common anti-pattern (`WS → Pub/Sub → Dataflow → Feature Store → Vertex Endpoint`) adds five failure modes, ~150ms, and — critically — **online/offline feature skew**, where live features drift from research features and live P&L stops matching the backtest.

**The single most important rule: the same feature code runs in research and live.** Pure functions over a buffer; research calls them over history, live calls them over a rolling window. No reimplementation.

**Latency is a non-issue.** Broker feeds are ~1 snapshot/sec, so you're quantised at ~1000ms before your code runs. End-to-end (~250ms–1.4s) is ~0.1% of a 15-minute hold. Optimise data quality, not speed. Use `asia-south1` and stop thinking about it.

---

## 11. Known limitations

**Read this before extending the work.**

1. **v1 uses only bar-derived features.** No order book, no option OI, no futures basis. `microstructure.py` and `institutional.py` are written but unused in the reported result. This is the most likely reason 15-minute horizons underperform while 60-minute works — short-horizon information lives in microstructure we aren't recording yet.

2. **Results are top-5% signals from walk-forward folds, not a full backtest.** No daily trade cap, no slippage on fast moves, no regime filter. Treat the hit rates as an upper bound.

3. **Margins are thin.** ₹220–773/trade at 5 lots assumes 0.67% option slippage. If real fills are worse, even 40-point targets stop working.

4. **Modelled on spot, not the traded instrument.** Options P&L adds theta and IV effects not captured here. Validate on futures, execute on options.

5. **Edge is not stable.** One fold in nine showed zero edge. Expect losing quarters.

6. **Charge rates and expiry rules go stale.** Rates reflect post-Oct-2024 STT. SEBI has repeatedly revised weekly expiry structure (BANKNIFTY weeklies discontinued; NIFTY/SENSEX expiry weekdays moved more than once). Days-to-expiry is a dominant feature — pass the real exchange calendar, never infer from a weekday rule.

7. **Lot sizes change.** Pulled live from Dhan's scrip master (NIFTY 65, BANKNIFTY 30, SENSEX 20 as of Nov 2026). Re-check before trusting any ₹ figure.

---

## Disclaimer

Research code for educational purposes. Not investment advice. Trading index derivatives carries substantial risk of loss. Past backtested performance does not indicate future results — and the results here are explicitly *not* a validated trading strategy. Do not deploy capital on this without completing Phases 2–4 and forming your own independent judgement.

## License

MIT
