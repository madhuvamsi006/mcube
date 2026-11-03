#!/usr/bin/env bash
# Deploy the collector as a Cloud Run Job, triggered each trading morning.
#
# Why a Job and not a VM:
#   - billed only for the ~6.5h it runs, not 24h
#   - no OS to patch, no disk to manage
#   - Cloud Scheduler starts it; Cloud Run retries if the task dies
#
# Why a Job and not a Service:
#   A Service is request-driven and scales to zero when no request is in
#   flight. This process has no inbound requests -- it holds an OUTBOUND
#   websocket for hours. Jobs allocate CPU for the whole task, which is what
#   that needs.
#
# Task timeout is 24h max (ample for a 6h15m session).
# The filesystem is in-memory tmpfs, so the collector runs with --cloud and
# ships each parquet to GCS a minute after writing it.

set -euo pipefail

PROJECT="${GCP_PROJECT_ID:?set GCP_PROJECT_ID}"
REGION="${REGION:-asia-south1}"          # Mumbai -- nearest to NSE and Dhan
BUCKET="${GCS_BUCKET:?set GCS_BUCKET}"
JOB="mcube-collector"
SA="mcube-collector@${PROJECT}.iam.gserviceaccount.com"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT}/mcube/collector:latest"

echo "project=$PROJECT region=$REGION bucket=$BUCKET"

# --- one-time setup ---------------------------------------------------------
gcloud config set project "$PROJECT"

gcloud services enable \
  run.googleapis.com \
  cloudscheduler.googleapis.com \
  artifactregistry.googleapis.com \
  secretmanager.googleapis.com \
  storage.googleapis.com

gcloud artifacts repositories create mcube \
  --repository-format=docker --location="$REGION" \
  --description="mcube collector images" 2>/dev/null || echo "artifact repo exists"

gsutil mb -l "$REGION" "gs://${BUCKET}" 2>/dev/null || echo "bucket exists"

# Dedicated service account: the job needs to write objects and read secrets,
# nothing else. Do not reuse the default compute SA, which is far broader.
gcloud iam service-accounts create mcube-collector \
  --display-name="mcube collector" 2>/dev/null || echo "service account exists"

gcloud storage buckets add-iam-policy-binding "gs://${BUCKET}" \
  --member="serviceAccount:${SA}" --role="roles/storage.objectAdmin" >/dev/null

# --- secrets ----------------------------------------------------------------
# Credentials live in Secret Manager, never in the image or in env vars set at
# deploy time. Client id + PIN + TOTP seed together allow order placement, so
# only this service account gets access.
create_secret () {
  local name=$1 prompt=$2
  if ! gcloud secrets describe "$name" >/dev/null 2>&1; then
    echo ""
    read -rsp "$prompt: " value; echo ""
    printf '%s' "$value" | gcloud secrets create "$name" --data-file=- --replication-policy=automatic
  else
    echo "secret $name exists (use: gcloud secrets versions add $name --data-file=- to rotate)"
  fi
  gcloud secrets add-iam-policy-binding "$name" \
    --member="serviceAccount:${SA}" --role="roles/secretmanager.secretAccessor" >/dev/null
}

create_secret dhan-client-id   "Dhan client id"
create_secret dhan-pin         "Dhan PIN"
create_secret dhan-totp-secret "Dhan TOTP seed (base32)"

# --- build and deploy -------------------------------------------------------
gcloud builds submit --tag "$IMAGE" .

gcloud run jobs deploy "$JOB" \
  --image="$IMAGE" \
  --region="$REGION" \
  --service-account="$SA" \
  --task-timeout=25200s \
  --max-retries=3 \
  --memory=2Gi \
  --cpu=1 \
  --set-env-vars="GCS_BUCKET=${BUCKET},MCUBE_OUT=/tmp/live,TZ=Asia/Kolkata" \
  --set-secrets="DHAN_CLIENT_ID=dhan-client-id:latest,DHAN_PIN=dhan-pin:latest,DHAN_TOTP_SECRET=dhan-totp-secret:latest" \
  --args="--symbols,NIFTY,--strikes,10,--cloud"

# --- schedule ---------------------------------------------------------------
# 09:10 IST on weekdays, five minutes before the open so subscriptions are
# live by 09:15. The job exits itself shortly after the close.
#
# NSE trading holidays are NOT handled here: on a holiday the job starts,
# receives nothing, and exits. Cheap and harmless.
gcloud scheduler jobs create http "${JOB}-trigger" \
  --location="$REGION" \
  --schedule="10 9 * * 1-5" \
  --time-zone="Asia/Kolkata" \
  --uri="https://${REGION}-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/${PROJECT}/jobs/${JOB}:run" \
  --http-method=POST \
  --oauth-service-account-email="$SA" \
  2>/dev/null || echo "scheduler job exists -- update with: gcloud scheduler jobs update http ${JOB}-trigger --location=$REGION"

gcloud projects add-iam-policy-binding "$PROJECT" \
  --member="serviceAccount:${SA}" --role="roles/run.invoker" >/dev/null

cat <<EOF

Deployed.

  Test now:     gcloud run jobs execute $JOB --region=$REGION
  Logs:         gcloud run jobs executions list --job=$JOB --region=$REGION
  Data:         gsutil ls gs://${BUCKET}/mcube/live/
  Cost:         ~6.5h/day x 22 days, 1 vCPU + 2GiB  =>  roughly \$5-8/month

EOF
