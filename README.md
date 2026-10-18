# Public Release

Offline monocular SLAM using XFeat and LighterGlue, with OpenCV and Pangolin
visualization. No ROS required. Datasets are not included.

## Prepare Data

Place one camera's image sequence and configuration files under `data/`:

```text
data/sequence/
  images/
    000001.png
    000002.png
    ...
  system.yaml
  camera.yaml
```

- Supported images: PNG, JPG/JPEG, BMP and TIF/TIFF.
- Images are sorted by filename. Use zero-padded names in chronological order.
- Use a consistent image resolution. No timestamps or ground-truth poses are
  required for image-sequence input.

Create configuration files from the supplied templates:

```bash
cp config/param/vo_fast.yaml data/sequence/system.yaml
cp config/param/feto_cam.yaml data/sequence/camera.yaml
```

Edit `camera.yaml` with **your camera calibration**: `cam_width`, `cam_height`,
`cam_fx`, `cam_fy`, `cam_cx`, `cam_cy`, and distortion coefficients `cam_k1`,
`cam_k2`, `cam_p1`, `cam_p2`, `cam_k3`. Calibration must match the supplied images,
including any resizing or prior undistortion. Template values are not universal.
Adjust `system.yaml` for your sequence; shared defaults are in `config/param/`.

## Run With Docker

From the project root, on Linux x86-64 with Docker, `xauth`, and an active
X11/XWayland desktop:

```bash
docker build -t fetal-slam:public .
bash scripts/run_docker.sh python3 -m fetal_slam \
  data/sequence/system.yaml data/sequence/camera.yaml data/sequence \
  --data-mode image-sequence --image-subdir images
```

The launcher mounts `data/` read-only. The image includes the native bindings
and uses CPU inference with software OpenGL. See [Docker details](docker/README.md).

## Run Locally

```bash
conda env create -f environment.yml
conda activate fetal-slam
python -m pip install --upgrade "pip>=23,<26"
python -m pip install --no-deps -e .
```

Also install compatible **g2o and Pangolin Python bindings** for this environment;
the Conda recipe alone does not provide them. Docker includes the required patches.

```bash
python3 -m fetal_slam \
  data/sequence/system.yaml data/sequence/camera.yaml data/sequence \
  --data-mode image-sequence --image-subdir images
```

Append `--max-frames 120` for a short run. At normal sequence completion, a dialog
offers optional surface reconstruction and Open3D visualization; bounded runs
skip this dialog. The surface is fitted from sparse points, not multi-view stereo.


