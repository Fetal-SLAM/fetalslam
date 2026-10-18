# Project Layout

## Maintained Runtime

`src/` directly contains the maintained SLAM modules. `pyproject.toml` maps this
directory to the `fetal_slam` Python package, preserving relative imports and
the existing CLI. Run `python -m pip install --no-deps -e .` in the active local
environment before launching or testing, using pip 23 or newer;
setting `PYTHONPATH=src` is not enough.
Docker installs the package during its build.

The command-line flow is:

```text
python -m fetal_slam
  -> fetal_slam.__main__
  -> fetal_slam.slam.cli
  -> XFeatSlamOffline
```

The runtime algorithms are `local_mapping.py` and `loop_closing.py`. Their
background execution is owned by `workers.py`; there are no duplicate
`*_thread.py` implementations in the maintained package.

Dense reconstruction has one maintained path: `dense_job.py` asks at sequence
EOF and displays the result; `dense_map.py` performs surface sampling,
pose-only PBA and color fusion. Unity/ex-vivo use `plane_mesh.py` for RANSAC,
a four-corner regular grid, and point-constrained height smoothing. The older
PCA/Delaunay mode is an explicit alternative. `dense_config.py` validates defaults from
`config/param/dense_map.yaml`, with overrides from the selected system YAML's
`dense_map_config`. The obsolete convex-hull reconstruction path is removed from
the runtime.

The XFeat inference subset required at runtime lives in
`src/features`, including its license and model weights.

## Non-Runtime Code

Historical research scripts, old flat module copies, ROS utilities, upstream
training demos and historical tests are not distributed in this release.
Regression tests are under `tests`; local and Docker launchers are under
`scripts`. Model weights are included; datasets and reconstruction outputs are not.

## Configuration

`runtime_config.py` loads and validates `config/param/runtime.yaml`. Each runtime
instance owns its settings; profile overrides do not modify shared defaults.
`parameters.py` merges the selected VO profile and camera calibration. Dense
settings retain their separate validator in `dense_config.py`.

Runtime diagnostics use module loggers. Logging is configured on the
`fetal_slam` logger only, with no root-logger changes or default log file.
Meaningful geometry and concurrency comments, numerical safety checks, and
upstream license attribution are retained. Public map/frame accessors remain
available even when the present CLI does not call every accessor.
