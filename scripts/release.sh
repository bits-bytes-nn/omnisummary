#!/usr/bin/env bash
# Build, push and deploy the image of the pushed, CI-green HEAD of main. Usage: scripts/release.sh
# Refuses a dirty tree, an unpushed HEAD or a commit whose CI did not pass, so what runs in Lambda is a
# reviewed commit; the image carries that commit as a tag and a label. Needs the operator's local
# config/*.yaml and .env (they are baked in / read by deploy.py) and AWS credentials for the profile.
set -euo pipefail
cd "$(dirname "$0")/.."
# The account and region come from the config (aws.profile, aws.region), as for every other tool here: a shell's
# AWS_PROFILE is often the model-calling one and may be another account.
PROFILE="$(uv run python -c 'from shared import Config; print(Config.load().aws.profile)')"
REGION="$(uv run python -c 'from shared import Config; print(Config.load().aws.region)')"
[ -n "$PROFILE" ] || { echo "aws.profile is empty in config/config.yaml" >&2; exit 1; }

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
# The image that ships carries the operator's config, which CI never sees: the same import and config checks
# as the CI docker job, on this image, before anything is pushed.
check_image() {
  docker run --rm --network none "$@" --entrypoint python "${ECR}:${SHA}" -c "$IMAGE_CHECK"
}
IMAGE_CHECK="import entrypoints.digest_handler, entrypoints.visual_handler, entrypoints.threads_refresh_handler, entrypoints.comment_reply_handler
from pathlib import Path
from shared import Config
c = Config.load()
assert c.collectors.web_search.trend_searches, 'the image resolved to bare code defaults'
assert Path(c.pipeline.visual_character_reference).is_file(), 'persona sheet missing'
assert c.edition != 'invest' or (c.pipeline.threads_footer and c.pipeline.domain_rules), 'investing edition config'
print('image OK:', c.edition or 'ai', c.aws.region, c.pipeline.digest_model.value)"
check_image
if [ -f config/invest.yaml ]; then check_image -e OMNISUMMARY_EDITION=invest; fi
docker push "${ECR}:${SHA}"
docker push "${ECR}:latest"
DIGEST_IMAGE_REF="$(aws ecr describe-images --profile "$PROFILE" --region "$REGION" --repository-name "$REPO" \
  --image-ids imageTag="$SHA" --query 'imageDetails[0].imageDigest' --output text)"
export DIGEST_IMAGE_REF
echo "Deploying ${SHA:0:12} as ${DIGEST_IMAGE_REF}"
# The CDK CLI's SDK can fail to resolve an SSO profile the aws CLI still reads (near token expiry it drops the
# account silently), so cdk gets the credentials the aws CLI resolved for the profile. A failed export must abort
# (a bare eval of an empty string would not), and cdk must deploy to the account the image was pushed to.
(
  CREDS="$(aws configure export-credentials --profile "$PROFILE" --format env)"
  eval "$CREDS"
  unset AWS_PROFILE
  export AWS_REGION="$REGION"
  [ "$(aws sts get-caller-identity --query Account --output text)" = "$ACCOUNT" ] \
    || { echo "Exported credentials are not for account $ACCOUNT" >&2; exit 1; }
  npx cdk deploy --all -a "uv run python scripts/deploy.py" --require-approval never
)

# Read-only smoke check: every image function of the stage runs the image just pushed, and its update landed.
FUNCTIONS="$(aws lambda list-functions --profile "$PROFILE" --region "$REGION" --output text \
  --query "Functions[?starts_with(FunctionName, '${PROJECT_STAGE}-') && PackageType=='Image'].FunctionName")"
[ -n "$FUNCTIONS" ] || { echo "No ${PROJECT_STAGE}-* image functions found after deploy" >&2; exit 1; }
FAILED=0
for FN in $FUNCTIONS; do
  aws lambda wait function-updated-v2 --profile "$PROFILE" --region "$REGION" --function-name "$FN"
  read -r IMAGE STATE UPDATE <<<"$(aws lambda get-function --profile "$PROFILE" --region "$REGION" --function-name "$FN" \
    --query '[Code.ResolvedImageUri, Configuration.State, Configuration.LastUpdateStatus]' --output text)"
  if [ "${IMAGE##*@}" = "$DIGEST_IMAGE_REF" ] && [ "$STATE" = "Active" ] && [ "$UPDATE" = "Successful" ]; then
    echo "OK   $FN"
  else
    echo "FAIL $FN: image ${IMAGE##*@}, state $STATE, last update $UPDATE" >&2
    FAILED=1
  fi
done
[ "$FAILED" -eq 0 ] || { echo "Deployed functions do not all run ${DIGEST_IMAGE_REF}" >&2; exit 1; }
