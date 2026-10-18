import threading

import numpy as np
import open3d as o3d

from fetal_slam.dense_config import dense_settings
from fetal_slam.dense_map import DenseMap
from fetal_slam.plane_mesh import build_plane_mesh
from test_dense_runtime import dense_scene


def settings(**values):
    return dense_settings(dict(surface_mode='plane_grid', plane_grid_cells=48, **values))


def surface():
    x, y = np.meshgrid(np.linspace(-1, 1, 35), np.linspace(-0.6, 0.6, 25))
    return np.column_stack((x.ravel(), y.ravel(), np.full(x.size, 2.)))


def test_continuous_rectangle_rejects_outliers_and_fills_hole():
    points = surface()
    points = points[(np.abs(points[:, 0]) > .3) | (np.abs(points[:, 1]) > .3)]
    outliers = np.column_stack((np.linspace(-.8, .8, 30), np.zeros(30), np.full(30, 3.)))
    mesh, report = build_plane_mesh(np.vstack((points, outliers)), settings())
    vertices = np.asarray(mesh.vertices)
    ny, nx = report['grid_shape']
    assert len(vertices) == ny * nx
    assert len(mesh.triangles) == 2 * (ny - 1) * (nx - 1)
    assert len(mesh.cluster_connected_triangles()[1]) == 1
    assert report['inliers'] == len(points)
    np.testing.assert_allclose(vertices[:, 2], 2., atol=1e-8)
    assert np.any((np.abs(vertices[:, 0]) < .1) & (np.abs(vertices[:, 1]) < .1))
    assert report['supported_vertices'] < len(vertices)
    assert abs(mesh.get_surface_area() - 2.4) < 1e-6


def test_height_fit_preserves_relief_without_shrinking_footprint():
    points = surface()
    points[:, 2] += .04 * np.cos(np.pi * points[:, 0]) * np.cos(np.pi * points[:, 1])
    # All points in this fixture are valid relief, not RANSAC outliers.
    mesh, report = build_plane_mesh(points, settings(plane_distance_ratio=.1))
    assert report['inliers'] == len(points)
    v = np.asarray(mesh.vertices)
    expected = 2. + .04 * np.cos(np.pi * v[:, 0]) * np.cos(np.pi * v[:, 1])
    error = np.sqrt(np.mean((v[:, 2] - expected) ** 2))
    assert error < .01, error
    assert np.ptp(v[:, 2]) > .04
    np.testing.assert_allclose(np.ptp(v[:, :2], axis=0), [2., 1.2], atol=1e-5)
    corners = report['corners']
    edges = np.array([corners[1] - corners[0], corners[3] - corners[0]])
    lengths = np.linalg.norm(edges, axis=1)
    np.testing.assert_allclose(np.ptp(v @ (edges / lengths[:, None]).T, axis=0),
                               lengths, atol=1e-8)
    assert np.isfinite(v).all()


def test_plane_scale_rotation_and_cancel():
    points = surface()
    rotation = o3d.geometry.get_rotation_matrix_from_xyz((.4, .2, .3))
    transformed = points @ rotation.T * 7 + [3., -1., 4.]
    mesh, _ = build_plane_mesh(transformed, settings())
    restored = (np.asarray(mesh.vertices) - [3., -1., 4.]) @ rotation / 7
    np.testing.assert_allclose(restored[:, 2], 2., atol=1e-7)
    def cancelled():
        raise InterruptedError('cancel')
    try:
        build_plane_mesh(points, settings(), cancelled)
        raise AssertionError('Cancellation ignored')
    except InterruptedError:
        pass


def test_plane_grid_full_dense_and_pba():
    scene = dense_scene()
    scene.update(surface_mode='plane_grid', plane_grid_cells=48, device='cpu')
    result = DenseMap().run_snapshot(scene, threading.Event())
    assert len(result['points']) > 1000
    assert np.isfinite(result['points']).all()
    assert result['pba']['accepted_steps'] > 0
    assert result['pba']['residuals'][-1] < result['pba']['residuals'][0]


def test_plane_config_rejects_invalid_values():
    for value in ({'plane_grid_cells': 0}, {'plane_data_weight': 0},
                  {'plane_min_inlier_ratio': 1.1}, {'plane_distance_ratio': -1},
                  {'surface_mode': 'unknown'}, {'pba_min_overlap': .98},
                  {'pba_min_overlap': 1.01}, {'pba_backtracking_steps': 0}):
        try:
            dense_settings(value)
            raise AssertionError('Invalid plane config accepted')
        except ValueError:
            pass


def test_plane_rejects_collinear_input():
    points = np.column_stack((np.linspace(-1, 1, 30), np.zeros(30), np.ones(30)))
    try:
        build_plane_mesh(points, settings())
        raise AssertionError('Collinear input accepted')
    except ValueError as error:
        assert 'collinear' in str(error)


def test_pba_overlap_rejects_excessive_visibility_loss():
    import torch
    mapper = DenseMap()
    mask = torch.ones(1000, dtype=torch.bool)
    mask[:4] = False
    assert mapper.projection_overlap_valid(mask)
    mask[:6] = False
    assert not mapper.projection_overlap_valid(mask)
    assert not mapper.projection_overlap_valid(torch.ones(19, dtype=torch.bool))
    mapper.settings['pba_min_overlap'] = 1.
    assert not mapper.projection_overlap_valid(mask)


def test_unity_and_exvivo_enable_plane_grid():
    from pathlib import Path
    import yaml
    root = Path(__file__).resolve().parents[1]
    for name in ('vo_unity.yaml', 'vo_fast.yaml'):
        with (root / 'config' / 'param' / name).open() as stream:
            config = yaml.safe_load(stream)
        assert dense_settings(config['dense_map_config'])['surface_mode'] == 'plane_grid'
