# Release Validation

Checked on 2026-10-18. These are observed results, not accuracy benchmarks.

## Docker

- Built the Dockerfile successfully on Linux x86-64: Ubuntu 22.04, Python 3.10,
  CPU PyTorch 2.4.1, Kornia 0.7.3, Mesa software OpenGL.
- Native g2o/Pangolin and OpenCV/Open3D/Tk imports passed.
- All 86 regression tests passed inside the image.
- Real-data checks used `docker/smoke_test.py`, a non-root UID and read-only data.

| Sequence | Display | Processed Frames | Final State | Map Points | Nonblank Pangolin |
| --- | --- | ---: | --- | ---: | --- |
| Unity2 | Xvfb | 120 | TRACKING | 534 | Yes |
| Ex vivo | Host X11 via `scripts/run_docker.sh` | 120 | TRACKING | 200 | Yes |

The measured SLAM loops took approximately 110 seconds and 19 seconds,
respectively, excluding startup and image loading. Counts and timings can vary
with hardware, random geometry estimation and background worker scheduling.
The checks verify initialization, final tracking state, map population and
OpenGL pixel variation; they do not measure geometric accuracy.

Kornia emits an AMP deprecation warning. On the tested X11 desktop, Pangolin
reported a fallback framebuffer configuration; rendering still passed the
nonblank check. A silent console or artifact-free rendering on other display
drivers is not guaranteed.

## Additional Checks

- The wheel installed into a separate environment and its CLI, packaged YAML
  defaults and bundled weights were accessible outside the source checkout.
- Launchers resolved paths from another directory, including paths with spaces.
- Real LighterGlue weights loaded strictly without modifying Kornia's global
  defaults. Tests exercise image-size metadata, empty matches and rejection of
  missing metadata without silently falling back to MNN.
- The plane-relief smoothing fixture passed 20 repeated runs on the host.

## Not Validated Here

Full Unity/ex-vivo sequences, repeated global loops in these container runs,
trajectory accuracy, clinical use, real-time speed, GPU Docker inference,
Windows/macOS Docker display, and a fresh complete Conda installation were not
validated. The bounded data runs skip the end-of-sequence reconstruction dialog.
Dense geometry and CPU photometric optimization have regression coverage, but
the final Open3D result window was not exercised by these runs.
