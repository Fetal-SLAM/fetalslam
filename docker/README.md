# Docker

## Build

```bash
docker build -t fetal-slam:public .
```

The Ubuntu 22.04 / Python 3.10 image builds the `g2o` and `pangolin` Python
modules, installs XFeat/LighterGlue dependencies and runs the regression tests.
`docker/patch_bindings.py` adds required g2o edge fields and updates the old
bindings' pybind11 integration. `docker/finalize_bindings.py` supplies the legacy
g2o constant definitions needed by modern pybind11 and relinks the module.
Upstream repositories and revisions are recorded
in the Dockerfile. Build memory and time depend on the host; lower parallelism
with `--build-arg BUILD_JOBS=1` if necessary. This image targets Linux x86-64.

The default is CPU inference and Mesa software OpenGL. No NVIDIA runtime or
GPU support is configured, and CPU performance is not claimed to be real-time.

## Interactive Run

Place data under the repository's `data/standard_data` directory, then:

```bash
bash scripts/run_docker.sh unity
bash scripts/run_docker.sh unity data/standard_data/Images_unity_2 --max-frames 120
bash scripts/run_docker.sh exvivo
```

The host must have Docker access, `xauth`, and an active X11 or XWayland `DISPLAY`.
The launcher mounts a temporary authorization cookie, not the user's home
directory. It does not call `xhost +`, grant privileged mode, or expose the Docker
socket. Dataset files are mounted read-only. The host UID/GID is used inside the
container. Relative data paths refer to `/workspace` inside the container;
arbitrary host paths are not automatically mounted.

For Compose, supply `DISPLAY`, `XAUTHORITY` pointing to an accessible X11 cookie
file, and optionally `LOCAL_UID` / `LOCAL_GID` (defaults 1000). The shell launcher
is preferred because it prepares the cookie automatically. Native Wayland
without XWayland and Docker Desktop on Windows/macOS are not supported by these
display instructions.

## Headless Verification

```bash
docker run --rm --init \
  -v "$PWD/data:/workspace/data:ro" \
  fetal-slam:public xvfb-run -a -s '-screen 0 1280x1024x24' \
  bash scripts/run_slam.sh unity --max-frames 120
```

Xvfb creates a virtual display: it exercises the GUI code but shows no window
on the host desktop. Use `--max-frames` so a dense-reconstruction dialog cannot
wait indefinitely at EOF. A successful short run is not a full-sequence
tracking or reconstruction-quality evaluation.

For a stricter check of initialization, tracking, map points and nonblank
Pangolin rendering, replace the last command with:

```bash
python3 docker/smoke_test.py unity data/standard_data/Images_unity_2 --max-frames 120
```

Run this command inside the container under `xvfb-run`, as above. It deliberately
fails if the sequence is shorter than the requested frame count or ends without
the `TRACKING` state.
