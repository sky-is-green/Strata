#!/usr/bin/env bash
# Start (or resume) the Hugging Face upload detached from this terminal.  Log: ~/AI/v41-dev/hf/upload.log
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p ~/AI/v41-dev/hf
setsid nohup python3 -u "$here/hf_upload.py" "$@" >> ~/AI/v41-dev/hf/upload.log 2>&1 < /dev/null &
echo "started (pid $!) - tail -f ~/AI/v41-dev/hf/upload.log"
