#!/usr/bin/env bash
# Run both local→S3 syncs (X/RSSHub + YouTube) before the daily AWS digest.
#
# These sources block datacenter (Lambda) IPs, so they're collected here on a residential
# IP and parked in S3; the digest Lambda reads the parked files. Run on a local cron a few
# minutes BEFORE the digest EventBridge schedule. Each sync is independent — one failing
# does not abort the other (so a YouTube hiccup never drops the X refresh, and vice versa).
#
# Requirements:
#   - For X/RSSHub: the local RSSHub Docker container named `rsshub` (started here if stopped).
#   - AWS creds: AWS_PROFILE (config.yaml uses `research`) or S3_SYNC_ACCESS_KEY_ID/SECRET in .env.
#
# Cron example (18:30 KST daily, ~30 min before the 19:00 KST = 10:00 UTC digest) — `crontab -e`:
#   30 18 * * * /Users/youngmki/Projects/omnisummary/scripts/sync_all_to_s3.sh >> /tmp/omnisummary-sync.log 2>&1
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

# `--restart unless-stopped` does not restart a container that was stopped, and the X sync then fails
# every day. Start it and wait (bounded) until it answers.
if command -v docker >/dev/null && ! curl -fsS -o /dev/null --max-time 5 http://localhost:1200; then
  log "RSSHub not answering; starting container 'rsshub'..."
  docker start rsshub >/dev/null || log "docker start rsshub FAILED"
  for _ in $(seq 1 30); do
    curl -fsS -o /dev/null --max-time 5 http://localhost:1200 && break
    sleep 2
  done
fi

rc=0
for source in rsshub youtube; do
  log "Starting ${source} sync..."
  if "$UV" run python scripts/sync_to_s3.py "$source"; then
    log "${source} sync OK"
  else
    log "${source} sync FAILED (continuing)"
    rc=1
  fi
done

log "Sync run complete (exit ${rc})"
exit "$rc"
