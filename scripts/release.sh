#!/usr/bin/env bash
# Build, push and deploy the image of the pushed, CI-green HEAD of main. Usage: scripts/release.sh
# Refuses a dirty tree, an unpushed HEAD or a commit whose CI did not pass, so what runs in Lambda is a
# reviewed commit; the image carries that commit as a tag and a label. Needs the operator's local
# config/*.yaml and .env (they are baked in / read by deploy.py) and AWS credentials for the profile.
set -euo pipefail
cd "$(dirname "$0")/.."
PROFILE="${AWS_PROFILE_NAME:-research}"
REGION="$(uv run python -c 'from shared import Config; print(Config.load().aws.region)')"

[ -z "$(git status --porcelain)" ] || { echo "Working tree is not clean" >&2; exit 1; }
git fetch -q origin main
SHA="$(git rev-parse HEAD)"
[ "$SHA" = "$(git rev-parse origin/main)" ] || { echo "HEAD is not origin/main" >&2; exit 1; }
CI="$(gh run list --commit "$SHA" --workflow CI --json conclusion --jq '.[0].conclusion')"
[ "$CI" = "success" ] || { echo "CI for $SHA is '${CI:-missing}', not success" >&2; exit 1; }

ACCOUNT="$(aws sts get-caller-identity --profile "$PROFILE" --query Account --output text)"
PROJECT_STAGE="$(uv run python -c 'from shared import Config; c=Config.load(); print(f"{c.aws.project_name}-{c.aws.stage}")')"
REPO="${PROJECT_STAGE}-agent"
ECR="${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/${REPO}"

aws ecr get-login-password --profile "$PROFILE" --region "$REGION" | docker login --username AWS --password-stdin "${ECR%%/*}"
docker build --platform linux/amd64 --provenance=false --build-arg GIT_REVISION="$SHA" -t "${ECR}:${SHA}" -t "${ECR}:latest" .
docker push "${ECR}:${SHA}"
docker push "${ECR}:latest"
DIGEST_IMAGE_REF="$(aws ecr describe-images --profile "$PROFILE" --region "$REGION" --repository-name "$REPO" \
  --image-ids imageTag="$SHA" --query 'imageDetails[0].imageDigest' --output text)"
export DIGEST_IMAGE_REF
echo "Deploying ${SHA:0:12} as ${DIGEST_IMAGE_REF}"
AWS_PROFILE="$PROFILE" npx cdk deploy --all -a "uv run python scripts/deploy.py" --require-approval never
