#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ ${1:-} == --help || ${1:-} == -h ]]; then
  printf '%s\n' 'Usage: run_docker.sh [unity|exvivo] [dataset-under-data] [SLAM options...]' \
    'Build first: docker build -t fetal-slam:public .' \
    'Requires Linux X11 (or XWayland), DISPLAY and xauth. Data is mounted read-only.'
  exit 0
fi
: "${DISPLAY:?An X11/XWayland desktop DISPLAY is required}"
command -v xauth >/dev/null || { printf '%s\n' 'Install xauth on the host first.' >&2; exit 2; }
AUTH="$(mktemp)"
trap 'rm -f "$AUTH"' EXIT
xauth nlist "$DISPLAY" | sed 's/^..../ffff/' | xauth -f "$AUTH" nmerge -
if [[ ! -s "$AUTH" ]]; then
  printf '%s\n' 'No X11 authorization cookie found for DISPLAY.' >&2
  exit 2
fi
mkdir -p "$ROOT/data"
if [[ $# -eq 0 ]]; then set -- unity; fi
docker run --rm --init \
  --user "$(id -u):$(id -g)" \
  -e DISPLAY -e XAUTHORITY=/tmp/slam.xauth -e HOME=/tmp \
  -v "$AUTH:/tmp/slam.xauth:ro" \
  -v /tmp/.X11-unix:/tmp/.X11-unix:ro \
  -v "$ROOT/data:/workspace/data:ro" \
  "${SLAM_IMAGE:-fetal-slam:public}" "$@"
