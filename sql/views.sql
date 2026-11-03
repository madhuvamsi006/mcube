-- mcube BigQuery views over the collected tick data.
--
-- The external tables read parquet straight from GCS, so nothing is loaded or
-- duplicated: the collector writes a file and it is queryable a minute later.
--
-- Partition pruning matters here. Every view filters on `dt`, and any query
-- you write should too -- without a dt predicate BigQuery scans every day ever
-- collected, which gets expensive fast.
--
-- Placeholders ${PROJECT} and ${DATASET} are substituted by deploy/bigquery.sh.

-- ---------------------------------------------------------------------------
-- v_ticks: all packets with a real timestamp and instrument metadata attached.
-- ---------------------------------------------------------------------------
-- Security ids are REUSED across expiries, so the id alone is meaningless
-- historically. The universe file recorded on the same day is the only correct
-- mapping -- hence the join on dt as well as security_id.
CREATE OR REPLACE VIEW `${PROJECT}.${DATASET}.v_ticks` AS
SELECT
  TIMESTAMP_MICROS(DIV(t.recv_ns, 1000))              AS recv_ts,
  DATETIME(TIMESTAMP_MICROS(DIV(t.recv_ns, 1000)),
           'Asia/Kolkata')                            AS ist_ts,
  t.dt,
  t.code,
  t.security_id,
  t.segment,
  u.symbol,
  u.kind,
  u.strike,
  u.option_type,
  u.expiry,
  u.lot_size,
  t.ltp, t.ltq, t.atp, t.volume, t.oi,
  t.total_buy_qty, t.total_sell_qty,
  t.day_open, t.day_high, t.day_low, t.day_close,
  t.bid_px_1, t.bid_px_2, t.bid_px_3, t.bid_px_4, t.bid_px_5,
  t.bid_qty_1, t.bid_qty_2, t.bid_qty_3, t.bid_qty_4, t.bid_qty_5,
  t.ask_px_1, t.ask_px_2, t.ask_px_3, t.ask_px_4, t.ask_px_5,
  t.ask_qty_1, t.ask_qty_2, t.ask_qty_3, t.ask_qty_4, t.ask_qty_5
FROM `${PROJECT}.${DATASET}.ticks_ext` t
LEFT JOIN `${PROJECT}.${DATASET}.universe_ext` u
  ON CAST(t.security_id AS STRING) = CAST(u.security_id AS STRING)
 AND t.dt = u.dt;


-- ---------------------------------------------------------------------------
-- v_depth: microstructure, one row per full packet (code 8).
-- ---------------------------------------------------------------------------
-- Only code 8 carries the order book. OI arrives separately as code 5, which
-- is why those are split rather than coalesced.
CREATE OR REPLACE VIEW `${PROJECT}.${DATASET}.v_depth` AS
SELECT
  recv_ts, ist_ts, dt, security_id, symbol, kind, strike, option_type, expiry,
  ltp, volume,
  bid_px_1, ask_px_1, bid_qty_1, ask_qty_1,

  (bid_px_1 + ask_px_1) / 2                                   AS mid,
  ask_px_1 - bid_px_1                                         AS spread,
  SAFE_DIVIDE(10000 * (ask_px_1 - bid_px_1),
              (bid_px_1 + ask_px_1) / 2)                      AS spread_bps,

  -- Microprice: size-weighted mid, leaning toward the thinner side because
  -- that is the side that gets taken first. A better short-horizon fair value
  -- than the raw mid, and the gap between them predicts drift.
  SAFE_DIVIDE(bid_px_1 * ask_qty_1 + ask_px_1 * bid_qty_1,
              bid_qty_1 + ask_qty_1)                          AS microprice,

  -- Level-1 imbalance in [-1, 1]. Positive = bid-heavy.
  SAFE_DIVIDE(bid_qty_1 - ask_qty_1, bid_qty_1 + ask_qty_1)   AS imb_l1,

  -- Imbalance across all five levels -- less jumpy than L1 alone.
  SAFE_DIVIDE(
    (bid_qty_1 + bid_qty_2 + bid_qty_3 + bid_qty_4 + bid_qty_5)
      - (ask_qty_1 + ask_qty_2 + ask_qty_3 + ask_qty_4 + ask_qty_5),
    (bid_qty_1 + bid_qty_2 + bid_qty_3 + bid_qty_4 + bid_qty_5)
      + (ask_qty_1 + ask_qty_2 + ask_qty_3 + ask_qty_4 + ask_qty_5)
  )                                                           AS imb_l5,

  bid_qty_1 + bid_qty_2 + bid_qty_3 + bid_qty_4 + bid_qty_5
    + ask_qty_1 + ask_qty_2 + ask_qty_3 + ask_qty_4 + ask_qty_5 AS depth_total,

  -- Total order-book pressure as reported by the exchange, which is wider
  -- than the visible five levels.
  SAFE_DIVIDE(total_buy_qty - total_sell_qty,
              total_buy_qty + total_sell_qty)                 AS book_pressure
FROM `${PROJECT}.${DATASET}.v_ticks`
WHERE code = 8
  AND bid_px_1 > 0
  AND ask_px_1 > 0;


-- ---------------------------------------------------------------------------
-- v_bars_1m: 1-minute bars with depth and flow, per instrument.
-- ---------------------------------------------------------------------------
-- This is the join point with the research pipeline: the same shape as the
-- historical bars, plus the microstructure columns history does not have.
CREATE OR REPLACE VIEW `${PROJECT}.${DATASET}.v_bars_1m` AS
SELECT
  TIMESTAMP_TRUNC(recv_ts, MINUTE)                      AS ts,
  dt,
  security_id,
  ANY_VALUE(symbol)      AS symbol,
  ANY_VALUE(kind)        AS kind,
  ANY_VALUE(strike)      AS strike,
  ANY_VALUE(option_type) AS option_type,
  ANY_VALUE(expiry)      AS expiry,

  ARRAY_AGG(ltp ORDER BY recv_ts ASC  LIMIT 1)[OFFSET(0)]  AS open,
  MAX(ltp)                                                 AS high,
  MIN(ltp)                                                 AS low,
  ARRAY_AGG(ltp ORDER BY recv_ts DESC LIMIT 1)[OFFSET(0)]  AS close,

  -- `volume` is cumulative for the day, so the bar's own volume is the
  -- difference across the minute, not a sum.
  MAX(volume) - MIN(volume)                                AS bar_volume,
  MAX(volume)                                              AS cum_volume,

  AVG(mid)          AS avg_mid,
  AVG(microprice)   AS avg_microprice,
  AVG(spread)       AS avg_spread,
  AVG(spread_bps)   AS avg_spread_bps,
  AVG(imb_l1)       AS avg_imb_l1,
  AVG(imb_l5)       AS avg_imb_l5,
  AVG(book_pressure) AS avg_book_pressure,
  AVG(depth_total)  AS avg_depth,

  -- Microprice pulling away from the mid is the headline drift signal.
  AVG(microprice - mid) AS microprice_dev,

  COUNT(*) AS n_updates
FROM `${PROJECT}.${DATASET}.v_depth`
GROUP BY ts, dt, security_id;


-- ---------------------------------------------------------------------------
-- v_oi_1m: open interest per minute, from code 5 packets.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW `${PROJECT}.${DATASET}.v_oi_1m` AS
SELECT
  TIMESTAMP_TRUNC(recv_ts, MINUTE) AS ts,
  dt,
  security_id,
  ANY_VALUE(symbol)      AS symbol,
  ANY_VALUE(kind)        AS kind,
  ANY_VALUE(strike)      AS strike,
  ANY_VALUE(option_type) AS option_type,
  ANY_VALUE(expiry)      AS expiry,
  MAX(oi)                AS oi
FROM `${PROJECT}.${DATASET}.v_ticks`
WHERE code = 5 AND oi IS NOT NULL
GROUP BY ts, dt, security_id;


-- ---------------------------------------------------------------------------
-- v_chain_oi: strike-level OI flow -- the institutional-intent signal.
-- ---------------------------------------------------------------------------
-- Institutions express directional views by WRITING options, so a change in
-- OI at a strike is the most direct read on intent available on NSE. Call
-- writing (CE OI rising) is bearish; put writing (PE OI rising) is bullish.
--
-- This cannot be backfilled from any API -- it exists only because the
-- collector recorded it.
CREATE OR REPLACE VIEW `${PROJECT}.${DATASET}.v_chain_oi` AS
WITH per_strike AS (
  SELECT
    ts, dt, symbol, strike, option_type, expiry,
    oi,
    oi - LAG(oi, 5)  OVER (PARTITION BY dt, security_id ORDER BY ts) AS doi_5m,
    oi - LAG(oi, 15) OVER (PARTITION BY dt, security_id ORDER BY ts) AS doi_15m
  FROM `${PROJECT}.${DATASET}.v_oi_1m`
  WHERE kind = 'option'
)
SELECT
  ts, dt, symbol, expiry,
  SUM(IF(option_type = 'CE', oi, 0))      AS ce_oi,
  SUM(IF(option_type = 'PE', oi, 0))      AS pe_oi,
  SUM(IF(option_type = 'CE', doi_5m, 0))  AS d_ce_oi_5m,
  SUM(IF(option_type = 'PE', doi_5m, 0))  AS d_pe_oi_5m,
  SUM(IF(option_type = 'CE', doi_15m, 0)) AS d_ce_oi_15m,
  SUM(IF(option_type = 'PE', doi_15m, 0)) AS d_pe_oi_15m,

  SAFE_DIVIDE(SUM(IF(option_type = 'PE', oi, 0)),
              SUM(IF(option_type = 'CE', oi, 0)))     AS pcr_oi,

  -- Net writing bias in [-1, 1]: positive = put writing dominates (bullish),
  -- negative = call writing dominates (bearish).
  SAFE_DIVIDE(
    SUM(IF(option_type = 'PE', doi_5m, 0)) - SUM(IF(option_type = 'CE', doi_5m, 0)),
    ABS(SUM(IF(option_type = 'PE', doi_5m, 0))) + ABS(SUM(IF(option_type = 'CE', doi_5m, 0)))
  )                                                   AS writing_bias_5m
FROM per_strike
GROUP BY ts, dt, symbol, expiry;


-- ---------------------------------------------------------------------------
-- v_collection_health: did the collector actually work that day?
-- ---------------------------------------------------------------------------
-- Check this before trusting any day's data. A session that ran but silently
-- received little is the failure mode worth catching early, while it can still
-- be fixed -- the data itself cannot be re-recorded.
CREATE OR REPLACE VIEW `${PROJECT}.${DATASET}.v_collection_health` AS
SELECT
  dt,
  COUNT(*)                                   AS total_packets,
  COUNT(DISTINCT security_id)                AS instruments,
  MIN(ist_ts)                                AS first_packet_ist,
  MAX(ist_ts)                                AS last_packet_ist,
  COUNT(DISTINCT FORMAT_TIMESTAMP('%H:%M', recv_ts, 'Asia/Kolkata')) AS minutes_covered,
  -- A full session is 375 minutes. Materially below that means gaps.
  ROUND(100 * COUNT(DISTINCT FORMAT_TIMESTAMP('%H:%M', recv_ts, 'Asia/Kolkata')) / 375.0, 1)
                                             AS session_coverage_pct,
  COUNTIF(code = 8)                          AS depth_packets,
  COUNTIF(code = 5)                          AS oi_packets
FROM `${PROJECT}.${DATASET}.v_ticks`
GROUP BY dt
ORDER BY dt DESC;
