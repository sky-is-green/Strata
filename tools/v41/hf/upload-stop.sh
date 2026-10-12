#!/usr/bin/env bash
# Pause the upload (the next upload-start.sh resumes at the first part not yet on the Hub).  Kills by PID only.
f=~/AI/v41-dev/hf/upload.lock
pid=$(cat "$f" 2>/dev/null)
if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null && grep -q hf_upload.py "/proc/$pid/cmdline" 2>/dev/null; then
    kill "$pid" && echo "stopped upload (pid $pid)"
else
    echo "no upload running"
fi
