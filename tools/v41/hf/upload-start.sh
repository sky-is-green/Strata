#!/usr/bin/env bash
# Start (or resume) the Hugging Face upload detached from this terminal.  Log: ~/AI/v41-dev/hf/upload.log
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p ~/AI/v41-dev/hf
# hf_xet's adaptive controller ramped to ~27 parallel 64 MB uploads on a ~1.85 MB/s link: requests timed out and were
# retried (~1/3 of the bandwidth wasted, 2026-10-12).  A fixed small concurrency keeps every request well inside its timeout.
export HF_XET_FIXED_UPLOAD_CONCURRENCY="${HF_XET_FIXED_UPLOAD_CONCURRENCY:-3}"
setsid nohup python3 -u "$here/hf_upload.py" "$@" >> ~/AI/v41-dev/hf/upload.log 2>&1 < /dev/null &
echo "started (pid $!) - tail -f ~/AI/v41-dev/hf/upload.log"
