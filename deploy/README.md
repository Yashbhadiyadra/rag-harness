# Deploy runbook - Cloud Run (public demo)

One-time GCP setup for the RAG harness public demo. You (the owner)
run these commands manually the first time; after that the release
workflow (`.github/workflows/release.yml`, wired in a follow-up
commit) handles image build and deploy on every tagged release.

See [`ADR-0010`](../docs/adr/ADR-0010-cloud-run-and-persistence.md)
for the design decisions this runbook implements.

**Nothing here is executed automatically.** Every command is one you
paste into your own shell and review before running. Anything that
changes state is called out explicitly.

## Authentication (required for non-demo deployments)

This runbook deploys the **public demo**, which runs with API
authentication OFF - it is protected by per-IP rate limiting and a
daily cap (ADR-0010). Any non-demo deployment must enable auth before
the service is exposed:

- Set `API_AUTH_ENABLED=true`.
- Generate a key hash with `rag-harness hash-key` and put it in
  `API_KEYS` (comma-separated for multiple keys). Store the raw keys in
  Secret Manager and hand them only to clients; the manifest holds only
  hashes.
- Enabling auth with an empty `API_KEYS` fails at startup by design, so
  a service can never come up "authenticated" while accepting every
  request.

See [`ADR-0023`](../docs/adr/ADR-0023-api-authentication.md).

## Prerequisites

- `gcloud` CLI installed and authenticated (`gcloud auth login`).
- A billing account you own (needed to enable APIs; the `$10/month`
  ceiling is enforced by a Cloud Billing budget set up in Step 6).
- A GitHub repo containing this codebase (for the WIF setup in Step 5).

## Variables

Set these once at the top of your shell session and reuse them
throughout. Substitute your own values first.

```bash
export PROJECT_ID="rag-harness-demo"       # dedicated project, clean billing
export REGION="us-central1"                # cheapest region, all services
export ARTIFACT_REPO="rag-harness"         # Artifact Registry repo name
export SERVICE_NAME="rag-harness"          # Cloud Run service name
export RUNTIME_SA="rag-harness-runtime"    # runtime service account (Secret Manager reader)
export DEPLOY_SA="rag-harness-deployer"    # deploy SA, GitHub Actions impersonates this
export WIF_POOL="github"                   # Workload Identity pool
export WIF_PROVIDER="github-actions"       # WIF provider inside the pool
export GITHUB_REPO="Yashbhadiyadra/rag-harness"   # owner/repo
export BILLING_ACCOUNT_ID="ABCDEF-012345-ABCDEF"  # from `gcloud billing accounts list`
```

---

## Step 1 - Create the project and enable APIs

**State change:** creates a new GCP project and links it to your
billing account.

```bash
gcloud projects create "$PROJECT_ID" --name="RAG Harness demo"
gcloud billing projects link "$PROJECT_ID" --billing-account="$BILLING_ACCOUNT_ID"
gcloud config set project "$PROJECT_ID"

gcloud services enable \
    run.googleapis.com \
    artifactregistry.googleapis.com \
    secretmanager.googleapis.com \
    cloudbuild.googleapis.com \
    iamcredentials.googleapis.com \
    iam.googleapis.com
```

Sanity check:

```bash
gcloud services list --enabled --filter="name:(run OR artifactregistry OR secretmanager OR iamcredentials)"
```

---

## Step 2 - Artifact Registry repo

**State change:** creates a Docker-format repo to hold the runtime
image.

```bash
gcloud artifacts repositories create "$ARTIFACT_REPO" \
    --location="$REGION" \
    --repository-format=docker \
    --description="RAG harness runtime images"
```

Full image reference format:

```
${REGION}-docker.pkg.dev/${PROJECT_ID}/${ARTIFACT_REPO}/${SERVICE_NAME}:<tag>
```

---

## Step 3 - Store the OpenAI key in Secret Manager

**State change:** creates a secret and adds the first version.

```bash
# Never echo the key. Paste it into stdin when prompted, then Ctrl-D.
gcloud secrets create openai-api-key --replication-policy=automatic
gcloud secrets versions add openai-api-key --data-file=-
```

To rotate later:

```bash
gcloud secrets versions add openai-api-key --data-file=-
# Cold-started containers will pick up the new `latest` version.
# Running containers keep the old value until the next start.
```

---

## Step 4 - Runtime service account

**State change:** creates the SA the Cloud Run container runs as, and
grants it the minimum roles: read the OpenAI secret, pull images.

```bash
gcloud iam service-accounts create "$RUNTIME_SA" \
    --display-name="RAG harness runtime"

RUNTIME_SA_EMAIL="${RUNTIME_SA}@${PROJECT_ID}.iam.gserviceaccount.com"

# Read the OPENAI_API_KEY secret. Scoped to just this secret.
gcloud secrets add-iam-policy-binding openai-api-key \
    --member="serviceAccount:${RUNTIME_SA_EMAIL}" \
    --role="roles/secretmanager.secretAccessor"

# Pull the runtime image from Artifact Registry.
gcloud artifacts repositories add-iam-policy-binding "$ARTIFACT_REPO" \
    --location="$REGION" \
    --member="serviceAccount:${RUNTIME_SA_EMAIL}" \
    --role="roles/artifactregistry.reader"
```

---

## Step 5 - Workload Identity Federation for GitHub Actions

**Why:** so the release workflow can push images and deploy without
storing a long-lived service-account JSON key in GitHub Secrets.

**State change:** creates a WIF pool, a provider (GitHub OIDC), a
deploy service account, and IAM bindings.

```bash
PROJECT_NUMBER=$(gcloud projects describe "$PROJECT_ID" --format="value(projectNumber)")

# 1. WIF pool
gcloud iam workload-identity-pools create "$WIF_POOL" \
    --location=global \
    --display-name="GitHub Actions"

# 2. OIDC provider bound to the pool. `attribute-condition` limits which
# GitHub repos can obtain tokens. This is the tenant-isolation step.
gcloud iam workload-identity-pools providers create-oidc "$WIF_PROVIDER" \
    --location=global \
    --workload-identity-pool="$WIF_POOL" \
    --display-name="GitHub Actions OIDC" \
    --issuer-uri="https://token.actions.githubusercontent.com" \
    --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository,attribute.ref=assertion.ref" \
    --attribute-condition="assertion.repository == '${GITHUB_REPO}'"

# 3. Deploy service account: this is what GitHub Actions impersonates.
gcloud iam service-accounts create "$DEPLOY_SA" \
    --display-name="RAG harness deploy (GitHub Actions)"

DEPLOY_SA_EMAIL="${DEPLOY_SA}@${PROJECT_ID}.iam.gserviceaccount.com"

# 4. Roles the deploy SA needs:
#    - push images to Artifact Registry
#    - deploy Cloud Run revisions
#    - act as the runtime SA (so `--service-account` on run deploy works)
for role in \
    roles/artifactregistry.writer \
    roles/run.admin \
    roles/iam.serviceAccountUser
do
    gcloud projects add-iam-policy-binding "$PROJECT_ID" \
        --member="serviceAccount:${DEPLOY_SA_EMAIL}" \
        --role="$role"
done

# 5. Allow the GitHub repo (via WIF) to impersonate the deploy SA.
gcloud iam service-accounts add-iam-policy-binding "$DEPLOY_SA_EMAIL" \
    --role=roles/iam.workloadIdentityUser \
    --member="principalSet://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${WIF_POOL}/attribute.repository/${GITHUB_REPO}"

echo "WIF provider resource name (put this in GitHub Actions):"
echo "projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${WIF_POOL}/providers/${WIF_PROVIDER}"
echo "Deploy SA email (put this in GitHub Actions):"
echo "${DEPLOY_SA_EMAIL}"
```

The release workflow in a follow-up commit will consume these two
values via `google-github-actions/auth@v2`.

---

## Step 6 - Cloud Billing budget with alerts

**State change:** creates a $10/month budget with email alerts at
50%, 90%, and 100%.

This is the belt to the daily-cap suspenders. The daily cap prevents
runaway spend within a day; the budget alerts prevent the demo from
silently drifting past the ceiling over a month.

```bash
gcloud billing budgets create \
    --billing-account="$BILLING_ACCOUNT_ID" \
    --display-name="RAG harness demo ($10/month)" \
    --budget-amount=10USD \
    --threshold-rule=percent=0.5 \
    --threshold-rule=percent=0.9 \
    --threshold-rule=percent=1.0 \
    --filter-projects="projects/${PROJECT_ID}"
```

Alert delivery goes to the billing account's admin email by default.
To add other recipients (e.g. a personal address), attach a Pub/Sub
topic and a Cloud Function, which is out of scope for this runbook; the
default email is enough for a one-owner demo.

---

## Step 7 - First manual deploy (sanity check)

The first deploy is done by hand so you can verify each piece works
before the release workflow starts doing it on every tag. This step
assumes you've already built and pushed an image.

Build and push locally:

```bash
IMAGE_TAG="${REGION}-docker.pkg.dev/${PROJECT_ID}/${ARTIFACT_REPO}/${SERVICE_NAME}:v0.1.0-manual"

# Authenticate Docker to Artifact Registry (once per shell).
gcloud auth configure-docker "${REGION}-docker.pkg.dev"

# Requires local chroma_db/ (see the top-level Makefile).
make docker-build
docker tag rag-harness:local "$IMAGE_TAG"
docker push "$IMAGE_TAG"
```

Substitute placeholders in the manifest and apply:

```bash
sed -e "s|__PROJECT_ID__|${PROJECT_ID}|g" \
    -e "s|__IMAGE_TAG__|${IMAGE_TAG}|g" \
    deploy/cloud-run.yaml \
    | gcloud run services replace - --region="$REGION"

# Allow unauthenticated invocations for the public demo.
gcloud run services add-iam-policy-binding "$SERVICE_NAME" \
    --region="$REGION" \
    --member=allUsers \
    --role=roles/run.invoker
```

Grab the URL and probe it:

```bash
URL=$(gcloud run services describe "$SERVICE_NAME" --region="$REGION" --format='value(status.url)')
echo "$URL"

curl -sf "${URL}/health"
curl -sf "${URL}/ready" | jq
curl -sf -X POST "${URL}/query" -H 'content-type: application/json' \
     -d '{"question":"What is a Pod?"}' | jq
```

Expected:
- `/health` → `{"status":"ok"}`
- `/ready` → 200 with `chromadb: ok`, `openai_api_key: present`.
- `/query` → 200 with an `answer`, at least one `sources` entry, a
  populated `trace`, non-zero `cost_usd` and `latency_ms`.

If any step fails, the runbook is wrong or the setup is wrong. Fix
before wiring the release workflow.

---

## Teardown

To remove everything the runbook created:

```bash
gcloud projects delete "$PROJECT_ID"
```

Deleting the project removes the service, images, secrets, budgets,
IAM bindings, WIF pool, and service accounts in one shot. The
project stays in "pending deletion" for 30 days; you can undelete
during that window if you change your mind.
