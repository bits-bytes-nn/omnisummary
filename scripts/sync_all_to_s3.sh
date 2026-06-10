#!/usr/bin/env bash
# Run both local→S3 syncs (X/RSSHub + YouTube) before the daily AWS digest.
#
# These sources block datacenter (Lambda) IPs, so they're collected here on a residential
# IP and parked in S3; the digest Lambda reads the parked files. Run on a local cron a few
# minutes BEFORE the digest EventBridge schedule. Each sync is independent — one failing
# does not abort the other (so a YouTube hiccup never drops the X refresh, and vice versa).
#
# Requirements:
#   - For X/RSSHub: the local RSSHub Docker container must be up (http://localhost:1200).
#   - AWS creds: AWS_PROFILE (config.yaml uses `research`) or S3_SYNC_ACCESS_KEY_ID/SECRET in .env.
#
# Cron example (07:50 KST daily, ~10 min before a 08:00 digest) — `crontab -e`:
#   50 7 * * * /Users/youngmki/Projects/omnisummary/scripts/sync_all_to_s3.sh >> /tmp/omnisummary-sync.log 2>&1
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR" || exit 1

: "${AWS_PROFILE:=research}"
export AWS_PROFILE

# cron runs with a minimal PATH that usually lacks uv (installed in ~/.local/bin or Homebrew).
# Prepend the common install dirs, then resolve an absolute uv path so the cron job doesn't
# die with "uv: command not found".
export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"
UV="$(command -v uv || true)"
if [ -z "$UV" ]; then
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] uv not found on PATH ($PATH) — aborting" >&2
  exit 127
fi

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

rc=0
for sync in sync_rsshub_to_s3 sync_youtube_to_s3; do
  log "Starting ${sync}..."
  if "$UV" run python "scripts/${sync}.py"; then
    log "${sync} OK"
  else
    log "${sync} FAILED (continuing)"
    rc=1
  fi
done

log "Sync run complete (exit ${rc})"
exit "$rc"
