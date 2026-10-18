"""Shared dense reconstruction configuration, loaded from config/param."""
from pathlib import Path
import sysconfig

import numpy as np
import yaml


def dense_settings(values=None):
    path = Path(__file__).resolve().parents[1] / 'config' / 'param' / 'dense_map.yaml'
    if not path.is_file():
        path = Path(sysconfig.get_path('data')) / 'share' / 'fetal-slam' / 'param' / 'dense_map.yaml'
    with path.open() as stream:
        settings = yaml.safe_load(stream)['dense_map_config']
    if values is not None:
        if not isinstance(values, dict) or set(values) - set(settings):
            raise ValueError('Invalid dense_map_config keys or format')
        settings.update(values)
    integers = dict(num_points=20, iterations=1, max_keyframes=0, margin=1,
                    min_views=2, stop_patience=1, min_sparse_points=3,
                    min_dense_points=1, min_pba_observations=6,
                    min_point_observations=2, gradient_kernel=3,
                    viewer_width=100, viewer_height=100,
                    plane_ransac_iterations=1, plane_grid_cells=2, plane_neighbors=1,
                    plane_smooth_iterations=0, pba_backtracking_steps=1)
    for name, minimum in integers.items():
        value = settings[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f'dense_map_config.{name} must be an integer >= {minimum}')
    positive = ('mesh_edge_factor', 'collinear_ratio', 'pba_damping',
                'max_rotation_step', 'max_translation_depth_ratio', 'gradient_scale',
                'plane_distance_ratio', 'plane_min_inlier_ratio',
                'plane_support_radius_factor', 'plane_data_weight', 'pba_min_overlap')
    nonnegative = ('residual_tolerance', 'sampling_min_depth', 'pba_min_depth', 'fusion_min_depth',
                   'plane_smooth_weight')
    for name in positive + nonnegative:
        value = settings[name]
        if (isinstance(value, bool) or not isinstance(value, (int, float)) or
                not np.isfinite(value) or value < 0 or (name in positive and value == 0)):
            raise ValueError(f'Invalid dense_map_config.{name}')
    if settings['max_keyframes'] == 1:
        raise ValueError('max_keyframes must be zero (all) or at least two')
    if settings['surface_mode'] not in ('pca', 'plane_grid'):
        raise ValueError('surface_mode must be pca or plane_grid')
    if settings['plane_min_inlier_ratio'] > 1:
        raise ValueError('plane_min_inlier_ratio must not exceed 1')
    if not 0.99 <= settings['pba_min_overlap'] <= 1:
        raise ValueError('pba_min_overlap must be between 0.99 and 1')
    if settings['iterations'] > 500:
        raise ValueError('iterations must not exceed 500')
    if settings['gradient_kernel'] not in (3, 5, 7):
        raise ValueError('gradient_kernel must be 3, 5 or 7')
    if settings['device'] not in ('auto', 'cpu', 'cuda'):
        raise ValueError('device must be auto, cpu or cuda')
    return settings
