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
