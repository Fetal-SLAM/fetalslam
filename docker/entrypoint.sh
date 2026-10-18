#!/usr/bin/env bash
set -euo pipefail
if [[ ${1:-} == unity || ${1:-} == exvivo ]]; then
  if [[ -z ${DISPLAY:-} ]]; then
    printf '%s\n' 'DISPLAY is required for Pangolin/OpenCV. Use scripts/run_docker.sh or xvfb-run for headless tests.' >&2
    exit 2
  fi
  exec bash /workspace/scripts/run_slam.sh "$@"
fi
exec "$@"
