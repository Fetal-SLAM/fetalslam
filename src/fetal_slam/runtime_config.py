"""Validated runtime defaults, shared by direct module use and the SLAM CLI."""

from functools import lru_cache
import logging
import math
from pathlib import Path
import sysconfig
from types import SimpleNamespace

import yaml


@lru_cache(maxsize=1)
def _defaults():
    path = Path(__file__).resolve().parents[2] / 'config/param/runtime.yaml'
    if not path.is_file():
        path = Path(sysconfig.get_path('data')) / 'share/fetal-slam/param/runtime.yaml'
    with path.open() as stream:
        return yaml.safe_load(stream)


def runtime_settings(section, overrides=None):
    defaults = _defaults()[section]
    if overrides is None:
        overrides = {}
    if not isinstance(overrides, dict):
        raise ValueError(f'{section} must be a mapping')
    unknown = set(overrides) - set(defaults)
    if unknown:
        raise ValueError(f'Unknown {section} settings: {sorted(unknown)}')
    values = {**defaults, **overrides}
    for name, value in values.items():
        default = defaults[name]
        if isinstance(default, bool):
            valid = isinstance(value, bool)
        elif isinstance(default, int):
            valid = type(value) is int and value >= (0 if name in ('loop_check_interval', 'loop_min_frame_gap', 'loop_cooldown_frames') else 1)
        elif isinstance(default, float):
            valid = type(value) in (int, float) and math.isfinite(value) and (value >= 0 if name == 'triangulation_distance_ratio' else value > 0)
        else:
            valid = isinstance(value, str)
        if not valid:
            raise ValueError(f'Invalid {section}.{name}: {value!r}')
        if isinstance(default, float) and 0 < default < 1 and value > 1:
            raise ValueError(f'{section}.{name} must be in (0, 1]')
        if name.endswith('confidence') and value >= 1:
            raise ValueError(f'{section}.{name} must be below 1')
    if section == 'logging' and values['level'] not in ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'):
        raise ValueError('logging.level must be DEBUG, INFO, WARNING, ERROR or CRITICAL')
    if section == 'viewer' and values['near_clip'] >= values['far_clip']:
        raise ValueError('viewer.near_clip must be below far_clip')
    if section == 'tracking' and values['image_margin'] >= 0.5:
        raise ValueError('tracking.image_margin must be below 0.5')
    if section == 'features':
        if values['homography_min_matches'] < 4:
            raise ValueError('features.homography_min_matches must be at least 4')
        if values['corner_block_size'] < 3 or values['corner_block_size'] % 2 == 0:
            raise ValueError('features.corner_block_size must be odd and at least 3')
        if values['corner_quality_decay'] <= 1:
            raise ValueError('features.corner_quality_decay must exceed 1')
    return SimpleNamespace(**values)


def configure_logging(settings):
    """Configure only this package; do not modify the application's root logger."""
    logger = logging.getLogger('fetal_slam')
    for handler in list(logger.handlers):
        if getattr(handler, '_fetal_slam_handler', False):
            logger.removeHandler(handler)
            handler.close()
    handler = logging.FileHandler(settings.file) if settings.file else logging.StreamHandler()
    handler._fetal_slam_handler = True
    handler.setFormatter(logging.Formatter('%(levelname)s %(name)s: %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(settings.level)
    logger.propagate = False
