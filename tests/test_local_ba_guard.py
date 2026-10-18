from types import SimpleNamespace

import numpy as np

from fetal_slam.optimizer import Optimizer
from test_local_map_recovery import make_recovery_scene


def scene(scale=1.0):
    closer, current, reference, _ = make_recovery_scene()
    optimizer = Optimizer(closer.optimizer.mK, 100, 100)
    current.pose[0, 3] = 0.3 * scale
    for key, point in reference.points.items():
        point.position *= scale
        point.erase_observation(2)
        point.add_observation(current.mnId)
        position = point.get_world_pos().reshape(3) + current.pose[:3, 3]
        projected = optimizer.mK @ position
        current.update_map_point(point, projected[:2] / projected[2], key)
    frames = {reference.mnId: reference, current.mnId: current}
    poses = {key: frame.get_pose().copy() for key, frame in frames.items()}
    points = {key: point.get_world_pos().reshape(3).copy() for key, point in reference.points.items()}
    return optimizer, frames, reference.points, poses, points


def test_local_ba_rejects_scale_collapse_despite_identical_reprojections():
    for scale in (0.001, 1.0, 1000.0):
        optimizer, frames, points, before_poses, before_points = scene(scale)
        after_poses = {key: pose.copy() for key, pose in before_poses.items()}
        for pose in after_poses.values():
            pose[:3, 3] *= 0.1
        after_points = {key: point * 0.1 for key, point in before_points.items()}
        for frame_id, pose in before_poses.items():
            for key, point in before_points.items():
                old = pose[:3, :3] @ point + pose[:3, 3]
                new = after_poses[frame_id][:3, :3] @ after_points[key] + after_poses[frame_id][:3, 3]
                np.testing.assert_allclose(old[:2] / old[2], new[:2] / new[2])
        report = optimizer.validate_local_ba_geometry(frames, points, before_poses, before_points, after_poses, after_points)
        assert not report['accepted']
        assert report['reason'] == 'median depth collapse'


def test_local_ba_rejects_partial_collapse_and_negative_depth():
    for factor, reason in ((0.1, 'partial depth collapse'), (-1.0, 'points moved behind camera')):
        optimizer, frames, points, poses, before = scene()
        after = {key: value.copy() for key, value in before.items()}
        for key in list(after)[:8]:
            after[key] *= factor
        report = optimizer.validate_local_ba_geometry(frames, points, poses, before, poses, after)
        assert report['reason'] == reason


def test_local_ba_rejects_flattening_and_cloud_contraction():
    for flatten in (True, False):
        optimizer, frames, points, poses, before = scene()
        after = {key: value.copy() for key, value in before.items()}
        for value in after.values():
            if flatten:
                value[2] = 3.7
            else:
                value[:2] *= 0.1
        report = optimizer.validate_local_ba_geometry(frames, points, poses, before, poses, after)
        assert report['reason'] == ('depth distribution flattened' if flatten else 'local point cloud contracted')


def test_local_ba_rejects_camera_baseline_collapse():
    optimizer, frames, points, poses, before = scene()
    after_poses = {key: value.copy() for key, value in poses.items()}
    for value in after_poses.values():
        value[:3, 3] *= 0.1
    report = optimizer.validate_local_ba_geometry(frames, points, poses, before, after_poses, before)
    assert report['reason'] == 'camera baseline collapse'


def test_one_flattened_view_does_not_veto_the_entire_window():
    optimizer, frames, points, poses, before = scene()
    angle = np.pi / 4
    poses[max(poses)][:3, :3] = np.array([
        [np.cos(angle), 0., np.sin(angle)], [0., 1., 0.],
        [-np.sin(angle), 0., np.cos(angle)],
    ])
    after = {key: value.copy() for key, value in before.items()}
    for value in after.values():
        value[2] = 3.7
    report = optimizer.validate_local_ba_geometry(frames, points, poses, before, poses, after)
    assert report['accepted']


def test_local_ba_accepts_small_corrections():
    optimizer, frames, points, poses, before = scene()
    after = {key: value * 0.97 for key, value in before.items()}
    report = optimizer.validate_local_ba_geometry(frames, points, poses, before, poses, after)
    assert report['accepted']


def test_local_ba_transaction_retains_both_poses_and_points():
    optimizer, frames, points, poses, before = scene()
    collapsed_poses = {key: value.copy() for key, value in poses.items()}
    for value in collapsed_poses.values():
        value[:3, 3] *= 0.1
    optimizer.bundle_adjustment = lambda *args: (collapsed_poses, {key: value * 0.1 for key, value in before.items()})
    tree = SimpleNamespace(get_all_temp_keyframes=lambda: list(frames.values()))
    output_poses, output_points = optimizer.local_bundle_adjustment(tree)
    assert optimizer.local_ba_rejections == 1
    for key in poses:
        np.testing.assert_array_equal(output_poses[key], poses[key])
        np.testing.assert_array_equal(frames[key].get_pose(), poses[key])
    for key in before:
        np.testing.assert_array_equal(output_points[key], before[key])
        np.testing.assert_array_equal(points[key].get_world_pos().reshape(3), before[key])


def test_ba_outputs_own_memory_after_g2o_optimizer_destruction():
    optimizer, frames, points, _, _ = scene()
    poses, optimized = optimizer.bundle_adjustment(frames, points)
    assert optimized
    for value in list(poses.values()) + list(optimized.values()):
        assert value.flags.owndata
        assert np.isfinite(value).all()


def test_local_ba_transaction_returns_accepted_correction():
    optimizer, frames, points, poses, before = scene()
    corrected = {key: value * 0.97 for key, value in before.items()}
    optimizer.bundle_adjustment = lambda *args: (poses, corrected)
    tree = SimpleNamespace(get_all_temp_keyframes=lambda: list(frames.values()))
    _, output = optimizer.local_bundle_adjustment(tree)
    assert optimizer.last_local_ba_validation['accepted']
    for key in before:
        np.testing.assert_array_equal(output[key], corrected[key])
        np.testing.assert_array_equal(points[key].get_world_pos().reshape(3), before[key])


def test_local_ba_rejects_nonfinite_or_incomplete_output():
    optimizer, frames, points, poses, before = scene()
    invalid = {key: value.copy() for key, value in before.items()}
    invalid[min(invalid)][0] = np.nan
    report = optimizer.validate_local_ba_geometry(frames, points, poses, before, poses, invalid)
    assert report['reason'] == 'non-finite geometry'
    report = optimizer.validate_local_ba_geometry(frames, points, poses, before, {}, before)
    assert report['reason'] == 'incomplete camera solution'
