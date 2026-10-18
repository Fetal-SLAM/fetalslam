# Public Release

Fetal SLAM is an experimental offline monocular SLAM pipeline for image
sequences. The Python package is `fetal_slam` in `src/fetal_slam`.

## What Runs

- XFeat feature extraction and XFeat-trained **LighterGlue**, a lightweight
  LightGlue variant, for descriptor matching. This is not the original
  SuperPoint-trained LightGlue model. Matching is followed by geometric checks.
- Shi-Tomasi points and homography-based propagation also participate in tracking
  and map construction. Not every map observation comes directly from LightGlue.
- Background local mapping and global loop closing, plus relocalization attempts
  after rejected or unclear frames. Recovery is not guaranteed.
- OpenCV tracking images and a Pangolin sparse-map viewer; ROS/RViz is not used.
- Optional surface reconstruction and pose-only photometric optimization after
  normal sequence completion, followed by an Open3D result window.

The system is monocular: map scale is arbitrary, not a metric measurement.
This is research software, not a validated clinical system. Tests do not prove
accuracy, real-time performance or long-sequence stability on unseen data.

## Docker

On Linux x86-64 with a Docker daemon and an X11/XWayland desktop:

```bash
docker build -t fetal-slam:public .
bash scripts/run_docker.sh unity
bash scripts/run_docker.sh exvivo --max-frames 120
```

Supply datasets in the layout below before launching. The image builds g2opy
and Pangolin Python bindings from pinned revisions and includes compatibility
patches required by this SLAM implementation. The default image uses CPU PyTorch
and software OpenGL; GPU acceleration is not configured. Compilation can take
several minutes. The host launcher requires `xauth` and a valid `DISPLAY`.
See [Docker](docker/README.md) for display setup and headless tests.

## Local Environment

```bash
conda env create -f environment.yml
conda activate fetal-slam
```

The Conda recipe installs Python dependencies, **not** the native g2o/Pangolin
bindings. Install compatible bindings for the active interpreter, or use Docker.
An arbitrary `g2o` package may not expose the edge fields required here; the
Docker build documents the exact source revisions and patches. The complete
Conda environment has not been recreated from scratch for this release.

```bash
bash scripts/run_unity.sh
bash scripts/run_unity.sh data/standard_data/Images_unity_2 --max-frames 120
bash scripts/run_exvivo.sh
```

The launchers use `python3` from the active environment, overridable with
`PYTHON_BIN`. They resolve relative dataset paths from the project root even
when invoked from another directory. Direct invocation from the project root:

```bash
PYTHONPATH=src python3 -m fetal_slam \
  config/param/vo_unity.yaml config/param/feto_cam.yaml \
  data/standard_data/Images_unity_1 --data-mode image-sequence --image-subdir images
```

## Data And Parameters

Datasets are not included or automatically downloaded:

```text
data/standard_data/Images_unity_1/images/
data/standard_data/Images_unity_2/images/
data/standard_data/ex_vivo_data/undist_image/
```

Unity uses `vo_unity.yaml` with `feto_cam.yaml`; ex vivo uses `vo_fast.yaml`
with `feto_cam_exvivo.yaml`. These calibrations are dataset-specific, not generic
camera defaults. Use calibration and image coordinates appropriate to your data.

Shared runtime defaults are in `config/param/runtime.yaml`; override sections
through `runtime_config` in the selected VO YAML. LighterGlue's match acceptance
setting is `runtime_config.features.lightglue_min_confidence` (default 0.1),
not the old cosine-similarity threshold. There is no silent MNN fallback.
`system_config` and `tracker_config` remain top-level profile sections.

Dense defaults are in `config/param/dense_map.yaml`, overridden by the profile's
`dense_map_config`. Current Unity settings request 300,000 samples and MSE-change
tolerance 0.001; ex vivo requests 1,000,000 samples and tolerance 0.00001.
Both cap PBA at 500 iterations, with 10 consecutive stable changes as an early
stop condition. Requested samples are not the final visible point count.

## Reconstruction Limits

The plane-grid mode fits a dominant plane, creates a rectangular grid, adjusts
vertex heights using nearby sparse points, smooths it, then samples and colors
the surface. It fills unobserved portions of that rectangle by assumption.
This is **not** multi-view stereo depth estimation. PBA optimizes camera poses
with fixed sampled 3D points, not joint point/pose bundle adjustment.

At normal EOF the workers finish and a Tk dialog asks whether to reconstruct.
Accepting opens the final result in Open3D. Early viewer closure or an explicit
`--max-frames` skips this dialog. Reconstruction does not overwrite the live map
or automatically export a point cloud. Logs and third-party diagnostics may
still be produced; this is not a promise of zero filesystem writes.

## Tests And Publication

```bash
PYTHONPATH=src python3 -m pytest -q
```

Tests cover geometry guards, repeated-loop bookkeeping, worker behavior,
configuration, matcher integration and portable launchers. Some use stubs;
passing them is not an end-to-end accuracy evaluation.
See [validation results](docs/VALIDATION.md) for the completed Docker builds,
120-frame data runs and explicit limits of those checks.

[Project layout](docs/PROJECT_LAYOUT.md) describes the maintained modules.
[Publication notes](docs/PUBLIC_RELEASE.md) explain history-free exports.
Private launchers, logs, datasets and editor state are excluded. Existing
remote branches are not made private or scrubbed by publishing a new branch.

Third-party source and model attribution is retained under
`src/fetal_slam/features`. No project-wide license has been selected for the
project's own code; do not infer that the bundled third-party license covers
the entire repository.
