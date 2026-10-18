import threading
import time
from unittest.mock import patch

import cv2
import numpy as np

from fetal_slam.dense_job import dense_settings, run_dense_after_sequence, snapshot_dense_scene
from fetal_slam.dense_map import DenseMap
from test_local_map_recovery import make_recovery_scene


def dense_scene():
    y, x = np.mgrid[:96, :96]
    gray = np.clip(120 + 40 * np.sin(x / 5.) + 40 * np.cos(y / 7.), 0, 255).astype(np.uint8)
    image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    shifted = cv2.warpAffine(image, np.array([[1., 0., 4.], [0., 1., 0.]]), (96, 96))
    pose = np.eye(4)
    pose[0, 3] = 0.11
    xx, yy = np.meshgrid(np.linspace(-0.65, 0.65, 12), np.linspace(-0.65, 0.65, 12))
    points = np.column_stack((xx.ravel(), yy.ravel(), np.full(xx.size, 2.)))
    return {"intrinsic": np.array([[80., 0., 48.], [0., 80., 48.], [0., 0., 1.]]),
            "points": points, "keyframes": {17: {"pose": np.eye(4), "image": image},
                                              83: {"pose": pose, "image": shifted}},
            "num_points": 2000, "iterations": 8}


def test_headless_dense_surface_and_real_cpu_pba():
    scene = dense_scene()
    before_points = scene['points'].copy()
    before_pose = scene['keyframes'][83]['pose'].copy()
    mapper = DenseMap()
    with patch('torch.cuda.is_available', return_value=False):
        result = mapper.run_snapshot(scene, threading.Event())
    assert len(result['points']) == 2000
    assert np.isfinite(result['points']).all()
    assert np.isfinite(result['colors']).all()
    assert np.all((result['colors'] >= 0) & (result['colors'] <= 1))
    assert mapper.pba_report['fixed_pose_id'] == 17
    assert mapper.pba_report['accepted_steps'] > 0
    assert mapper.pba_report['residuals'][-1] < mapper.pba_report['residuals'][0]
    np.testing.assert_array_equal(result['poses'][17]['pose'], np.eye(4))
    np.testing.assert_array_equal(scene['points'], before_points)
    np.testing.assert_array_equal(scene['keyframes'][83]['pose'], before_pose)


def test_snapshot_does_not_alias_live_map():
    _, current, reference, optimizer = make_recovery_scene()
    snapshot = snapshot_dense_scene({reference.mnId: reference}, reference.points, optimizer.mK)
    snapshot['points'][:] = 0
    snapshot['keyframes'][reference.mnId]['image'][:] = 0
    snapshot['keyframes'][reference.mnId]['pose'][:] = 0
    assert np.any(reference.image)
    assert np.any(next(iter(reference.points.values())).get_world_pos())
    np.testing.assert_array_equal(reference.get_pose(), np.eye(4))


def test_post_sequence_no_does_not_copy_or_build():
    from unittest.mock import Mock
    factory = Mock()
    with patch('tkinter.Tk'), patch('tkinter.messagebox.askyesno', return_value=False), \
            patch('fetal_slam.dense_job.show_dense_result') as show:
        assert run_dense_after_sequence(factory, dense_settings()) is None
        factory.assert_not_called()
        show.assert_not_called()


def test_post_sequence_yes_builds_then_displays():
    scene = dense_scene()
    with patch('tkinter.Tk'), patch('tkinter.messagebox.askyesno', return_value=True), \
            patch('fetal_slam.dense_job.show_dense_result') as show, \
            patch('torch.cuda.is_available', return_value=False):
        result = run_dense_after_sequence(lambda: scene, dense_settings())
        assert len(result['points']) == 2000
        show.assert_called_once_with(result, dense_settings())


def test_dense_configuration_validation():
    assert dense_settings()['num_points'] == 1000000
    assert dense_settings()['iterations'] == 500
    assert dense_settings()['max_keyframes'] == 0
    for values in ({'max_keyframes': 1}, {'num_points': -1}, {'device': 'invalid'},
                   {'iterations': 0}, {'iterations': 501}, {'stop_patience': 0},
                   {'residual_tolerance': -1}, {'residual_tolerance': float('nan')},
                   {'mesh_edge_factor': float('nan')}, {'typo': 1}):
        try:
            dense_settings(values)
            raise AssertionError(f'Accepted invalid settings: {values}')
        except ValueError:
            pass


def test_dense_filter_overrides_are_used():
    for options in ({'min_sparse_points': 200}, {'min_dense_points': 3000},
                    {'sampling_min_depth': 3.0}, {'min_pba_observations': 3000}):
        scene = dense_scene()
        scene.update(options, device='cpu')
        try:
            DenseMap().run_snapshot(scene, threading.Event())
            raise AssertionError(f'Filter ignored: {options}')
        except ValueError:
            pass


def test_dense_viewer_uses_configured_dimensions():
    from fetal_slam.dense_job import show_dense_result
    result = {'points': np.zeros((2, 3)), 'colors': np.ones((2, 3))}
    with patch('open3d.visualization.draw_geometries') as draw:
        show_dense_result(result, {'viewer_width': 640, 'viewer_height': 480})
    assert draw.call_args.kwargs['width'] == 640
    assert draw.call_args.kwargs['height'] == 480


def test_extended_dense_settings_validation():
    for options in ({'pba_damping': 0}, {'max_rotation_step': -1},
                    {'gradient_kernel': 4}, {'viewer_width': 0},
                    {'sampling_min_depth': float('nan')}, {'collinear_ratio': True}):
        try:
            dense_settings(options)
            raise AssertionError(f'Invalid setting accepted: {options}')
        except ValueError:
            pass


def test_pba_consecutive_plateau_and_iteration_limit():
    for limit, expected, reason in ((500, 4, 'residual_plateau'), (2, 2, 'max_iterations')):
        scene = dense_scene()
        scene['keyframes'][83]['pose'] = np.eye(4)
        scene['keyframes'][17]['image'][:] = 128
        scene['keyframes'][83]['image'] = scene['keyframes'][17]['image'].copy()
        scene.update(iterations=limit, stop_patience=3, residual_tolerance=1e-5, device='cpu')
        result = DenseMap().run_snapshot(scene, threading.Event())
        assert result['pba']['iterations'] == expected
        assert result['pba']['stop_reason'] == reason


def test_pba_plateau_counter_resets_on_meaningful_change():
    mapper = DenseMap()
    assert mapper.count_stable_iterations(float('inf'), 1., 0) == 0
    assert mapper.count_stable_iterations(1., 1., 0) == 1
    assert mapper.count_stable_iterations(1., 0.5, 5) == 0
    assert mapper.count_stable_iterations(0.5, 1., 5) == 0


def test_sequence_eof_drains_workers_before_prompt_and_limits_skip():
    from unittest.mock import Mock
    from fetal_slam.slam import XFeatSlamOffline
    for limit in (None, 10):
        instance = XFeatSlamOffline.__new__(XFeatSlamOffline)
        instance.load_from_folder = False
        instance.images_list = []
        instance.initial_flag = True
        instance.viewer = Mock()
        instance.map = Mock()
        instance.kf_tree = Mock()
        instance.kf_tree.get_all_keyframes_for_loopclosing.return_value = {}
        instance.kf_tree.get_all_temp_keyframes.return_value = []
        instance.map.get_all_key_frames.return_value = []
        instance.dense_config = dense_settings()
        instance.stop_workers = Mock()
        def prompt(factory, settings):
            instance.stop_workers.assert_called_once()
            instance.viewer.close.assert_called_once()
        with patch('fetal_slam.slam.run_dense_after_sequence', side_effect=prompt) as ask, \
                patch('cv2.destroyAllWindows'):
            instance.run('', max_frames=limit)
        assert ask.call_count == (1 if limit is None else 0)


def test_dense_uses_snapshot_camera_and_settings():
    scene = dense_scene()
    scene.update(margin=7, min_views=2, device='cpu', mesh_edge_factor=8.)
    mapper = DenseMap()
    mapper.run_snapshot(scene, threading.Event())
    np.testing.assert_array_equal(mapper.intrinsic_matrix, scene['intrinsic'])
    assert (mapper.width, mapper.height) == (96, 96)
    assert mapper.margin == 7 and mapper.num_points == 2000
    assert mapper.device == 'cpu'


def test_dense_cancelled_before_mesh_and_bad_geometry():
    cancel = threading.Event()
    cancel.set()
    try:
        DenseMap().run_snapshot(dense_scene(), cancel)
        raise AssertionError('cancellation was ignored')
    except InterruptedError:
        pass
    scene = dense_scene()
    scene['points'][:, 1:] = 0
    try:
        DenseMap().run_snapshot(scene, threading.Event())
        raise AssertionError('collinear geometry was accepted')
    except ValueError:
        pass
