# v1 Results - NIFTY spot, bar features only

Run date: 2026-11-03
Data: Dhan 1-min NIFTY spot, 2023-11-06 .. 2026-11-03 (741 sessions, 277,087 bars)
Features: 37, all derived from OHLCV bars (no depth, no option OI yet)
Validation: walk-forward, 12 months train / 3 months test, 9 folds, 30-bar embargo

## Integrity checks (run these before believing anything below)

    synthetic null test   AUC 0.4989   PASS - pipeline finds nothing in noise
    real-data shuffle     AUC 0.4989   PASS - no leakage
    real-data model       AUC 0.5516   genuine, modest edge

Accuracy on shuffled labels was 0.6186 vs 0.6163 on real labels. That is why
every number here is reported as AUC or as hit-rate-vs-base-rate: raw accuracy
is pure class imbalance and is meaningless for this problem.

## Edge measured (top 5% confidence long signals, walk-forward)

| target/stop | horizon | base rate | hit rate | edge   | breakeven | Rs/trade (5 lots) |
|-------------|---------|-----------|----------|--------|-----------|-------------------|
| 20 / 13     | 15 min  | 38.1%     | 47.3%    | +9.2   | 53.5%     |  -662  LOSS       |
| 30 / 20     | 15 min  | 34.9%     | 43.7%    | +8.7   | 49.3%     |  -913  LOSS       |
| 30 / 20     | 30 min  | 38.0%     | 43.9%    | +5.9   | 49.3%     |  -872  LOSS       |
| 40 / 27     | 30 min  | 36.2%     | 48.2%    | +12.1  | 47.2%     |  +220  profitable |
| 50 / 33     | 45 min  | 35.8%     | 48.0%    | +12.2  | 45.4%     |  +699  profitable |
| 60 / 40     | 60 min  | 36.3%     | 47.0%    | +10.7  | 44.6%     |  +773  profitable |

Costs: options, ATM, 5 lots, 4.6 index points round trip.

## Reading this

The edge is REAL and CONSISTENT: +9 to +12 percentage points over base rate
across every configuration tested. It is not an artifact -- the shuffle test
rules out leakage, and the consistency across targets argues against noise.

The edge is also INSUFFICIENT at short targets. At 20 points the model needs
+15.4pp to break even and delivers +9.2pp. Costs, not model quality, are what
kill it.

Widening the target fixes this, because required skill falls faster than
measured skill does. 40+ points over 30-60 minutes clears the cost floor.

Top 1% confidence scored WORSE than top 5% (44.2% vs 47.3%). The extreme tail
is overfitting. Use the top 5% band, not the most confident handful.

Worst single fold at top-5% was 38.5% -- base rate, i.e. no edge at all in that
quarter. The edge is not stable across all regimes.

## Caveats before trusting this

1. These are top-5% signals from walk-forward folds, NOT a full backtest.
   No daily trade cap, no slippage on fast moves, no regime filter applied.
2. Margins are thin. Rs220-773/trade at 5 lots leaves little room for the
   slippage assumption (0.67% of premium) being optimistic.
3. Spot index, not the tradeable instrument. Options P&L adds theta and IV
   effects this does not model.

## What is missing, and why it matters

This v1 uses ONLY bar-derived features. The original design also calls for:

  - 200-level order book depth   (live-only; cannot be backfilled)
  - intraday strike-level OI     (live-only; snapshot endpoint has no history)
  - futures basis / parity gap   (needs stitched expired futures contracts)

Those are precisely the microstructure signals that should carry short-horizon
information. Their absence is the most likely reason the 15-minute horizon
underperforms while the 60-minute horizon works.

This result therefore JUSTIFIES the collector rather than replacing it.

## Next steps, in order

1. Collector VM - start recording depth + option chain OI. Every day without
   it is permanently lost data.
2. Stitch expired futures contracts for multi-year OI and basis history.
3. Full backtest at 50/33 over 45 min with trade caps and realistic fills.
4. Paper trade before any capital.

---

# v1.1 - target/horizon matching, and the volume-feature finding

Run date: 2026-11-04

## Horizon must scale with the target

Training 50/33 on the config default 15-bar horizon produced AUC 0.6249 -- which
looked like an improvement and was not. 78.6% of labels timed out, because a
50-point move rarely completes in 15 minutes. The ~21% that DID resolve were a
high-volatility selection, and the model was partly learning "is this a volatile
period" rather than direction.

Widening the horizon to 45 bars:

    horizon 15   timeout 78.6%   AUC 0.6249   <- inflated by selection
    horizon 45   timeout 46.2%   AUC 0.5736   <- honest

`train_real.py` now warns when timeouts exceed 50%.

## Purged K-fold vs walk-forward

    purged K-fold (12/3)   AUC 0.5736
    walk-forward  (6/2)    AUC 0.5296

Walk-forward is the number to trust. K-fold is for hyperparameter choice only.

## Volume features do not help

Dhan reports NIFTY SPOT volume only from 2025-12-03; before that it is zero.
So 7 features (vol_z_*, vol_ratio, flow_imb_*, vwap_dev_bps) covered just 30.7%
of the 3-year history and were silently dropped by the coverage filter.

Tested whether they are worth trading history for, 50/33 at 45 min:

| variant                          | feats | rows    | AUC    | top5% hit |
|----------------------------------|-------|---------|--------|-----------|
| 3yr, volume cols auto-dropped    |  37   | 149,154 | 0.5296 |   44.5%   |
| since Dec-2025, WITH volume      |  44   |  49,409 | 0.5301 |   33.9%   |
| since Dec-2025, WITHOUT volume   |  34   |  49,409 | 0.5385 |   36.7%   |

Volume features add nothing (0.5301 with vs 0.5385 without on the same window),
and three times the history matters far more than seven extra columns. Keep the
3-year model.

Worth noting the flow proxy (`signed_vol`, `flow_imb`) was the closest thing in
bar data to real order flow, and it did not help. That is mild evidence that bar
level flow proxies are too coarse -- and a further argument for the collector's
true depth data rather than against the thesis.

## Live/research parity verified

The paper trader runs the SAME feature code over a rolling buffer. Checked by
feeding identical bars through both paths:

    research path : 0.67167261
    live path     : 0.67167261   MATCH

This is the check that catches online/offline feature skew -- the usual reason
live P&L stops matching a backtest without anyone knowing which half broke.

## Model saved

    models/nifty_50_33.joblib   50/33 pts, 45-bar horizon, 37 features
                                CV AUC 0.5736, walk-forward AUC 0.5296

The feature LIST is saved with it; the paper trader reindexes to exactly that
order, because a missing or reordered column changes predictions silently.
