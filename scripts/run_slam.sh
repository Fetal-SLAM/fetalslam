#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  printf '%s\n' \
    'Usage: run_slam.sh {unity|exvivo} [dataset-directory] [SLAM options...]' \
    'Relative dataset paths are resolved from the project root.' \
    'Default data: data/standard_data/Images_unity_1 or data/standard_data/ex_vivo_data' \
    'Example: run_slam.sh unity data/standard_data/Images_unity_2 --max-frames 120' \
    'Use PYTHON_BIN to select a Python interpreter; default: python3 from PATH.'
}

if [[ $# -eq 0 || "$1" == --help || "$1" == -h ]]; then
  usage
  exit 0
fi

case "$1" in
  unity)
    system_config=config/param/vo_unity.yaml
    camera_config=config/param/feto_cam.yaml
    dataset=data/standard_data/Images_unity_1
    image_subdir=images
    ;;
  exvivo)
    system_config=config/param/vo_fast.yaml
    camera_config=config/param/feto_cam_exvivo.yaml
    dataset=data/standard_data/ex_vivo_data
    image_subdir=undist_image
    ;;
  *) usage >&2; exit 2 ;;
esac
shift

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
  usage
  exit 0
fi

if [[ $# -gt 0 && "$1" != -* ]]; then
  dataset="$1"
  shift
fi

cd "$REPO_ROOT"
if [[ ! -d "$dataset" ]]; then
  printf 'Dataset directory not found: %s\n' "$dataset" >&2
  exit 2
fi

exec "${PYTHON_BIN:-python3}" -m fetal_slam \
  "$system_config" "$camera_config" "$dataset" \
  --data-mode image-sequence --image-subdir "$image_subdir" "$@"
