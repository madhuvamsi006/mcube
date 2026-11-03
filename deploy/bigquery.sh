#!/usr/bin/env bash
# Create BigQuery external tables + views over the collected tick data.
#
# EXTERNAL tables, not loaded tables: BigQuery reads the parquet straight from
# GCS, so the collector writes a file and it is queryable a minute later. No
# load step, no second copy, no storage billed twice.
#
# The trade-off is that external tables are slower than native ones on large
# scans. At ~20GB/year that does not matter. If it ever does, materialise the
# hot views into native partitioned tables with a scheduled query.
#
# Hive partitioning on dt=YYYY-MM-DD means a query filtering on `dt` reads only
# those files. ALWAYS filter on dt -- without it you scan every day collected.

set -euo pipefail

PROJECT="${GCP_PROJECT_ID:?set GCP_PROJECT_ID}"
BUCKET="${GCS_BUCKET:?set GCS_BUCKET}"
DATASET="${BQ_DATASET:-mcube}"
LOCATION="${REGION:-asia-south1}"
PREFIX="gs://${BUCKET}/mcube/live"

echo "project=$PROJECT dataset=$DATASET location=$LOCATION"
echo "source=$PREFIX"

gcloud config set project "$PROJECT" >/dev/null

# Dataset must sit in the same region as the bucket, or queries fail.
bq --location="$LOCATION" mk --dataset \
  --description="mcube collected market data (external over GCS)" \
  "${PROJECT}:${DATASET}" 2>/dev/null || echo "dataset exists"

tmp=$(mktemp -d)

make_def () {
  local name=$1 glob=$2
  cat > "${tmp}/${name}.json" <<JSON
{
  "sourceFormat": "PARQUET",
  "sourceUris": ["${PREFIX}/${glob}"],
  "hivePartitioningOptions": {
    "mode": "AUTO",
    "sourceUriPrefix": "${PREFIX}/"
  }
}
JSON
}

# Three separate tables because the schemas differ: ticks carry depth,
# universe is the id->strike map, chain is the REST snapshot with IV/greeks.
make_def ticks    "dt=*/ticks_*.parquet"
make_def universe "dt=*/universe.parquet"
make_def chain    "dt=*/chain_*.parquet"

for t in ticks universe chain; do
  bq rm -f -t "${PROJECT}:${DATASET}.${t}_ext" 2>/dev/null || true
  if bq mk --external_table_definition="${tmp}/${t}.json" \
       "${PROJECT}:${DATASET}.${t}_ext" 2>/dev/null; then
    echo "  created ${t}_ext"
  else
    # Expected until the collector has written at least one file of that kind.
    echo "  skipped ${t}_ext (no files under ${PREFIX}/ yet)"
  fi
done

# --- views ------------------------------------------------------------------
echo "creating views..."
sed -e "s/\${PROJECT}/${PROJECT}/g" -e "s/\${DATASET}/${DATASET}/g" \
    sql/views.sql > "${tmp}/views.sql"

if bq query --use_legacy_sql=false --location="$LOCATION" < "${tmp}/views.sql"; then
  echo "  views created"
else
  echo "  views failed -- usually means no data collected yet. Re-run after"
  echo "  the first session: ./deploy/bigquery.sh"
fi

rm -rf "$tmp"

cat <<EOF

Done. Try:

  -- did the collector work?
  bq query --use_legacy_sql=false \
    'SELECT * FROM \`${PROJECT}.${DATASET}.v_collection_health\` LIMIT 10'

  -- order book imbalance on the future, one day
  bq query --use_legacy_sql=false \
    'SELECT ist_ts, ltp, imb_l5, spread_bps
     FROM \`${PROJECT}.${DATASET}.v_depth\`
     WHERE dt = CURRENT_DATE("Asia/Kolkata") AND kind = "future"
     ORDER BY ist_ts DESC LIMIT 20'

  -- institutional option writing bias
  bq query --use_legacy_sql=false \
    'SELECT ts, pcr_oi, writing_bias_5m, d_ce_oi_5m, d_pe_oi_5m
     FROM \`${PROJECT}.${DATASET}.v_chain_oi\`
     WHERE dt = CURRENT_DATE("Asia/Kolkata")
     ORDER BY ts DESC LIMIT 20'

Always filter on dt. Without it every query scans every day ever collected.

EOF
