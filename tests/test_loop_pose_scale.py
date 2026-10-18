import numpy as np

from fetal_slam.optimizer import Optimizer
from test_local_map_recovery import make_recovery_scene


def test_loop_pose_solver_preserves_input_map_scale():
    camera = np.array([[400., 0., 360.], [0., 400., 360.], [0., 0., 1.]])
    optimizer = Optimizer(camera, 720, 720)
    rng = np.random.default_rng(42)
    points = rng.uniform([-1., -1., 3.], [1., 1., 6.], (60, 3))
    translation = np.array([0.3, -0.2, 0.4])
    projected = (camera @ (points + translation).T).T
    features = {i: xy[:2] / xy[2] for i, xy in enumerate(projected)}
    for scale in (0.1, 1.0, 10.0):
        world = {i: p * scale for i, p in enumerate(points)}
        snapshot = {i: p.copy() for i, p in world.items()}
        count, rotation, recovered = optimizer.pose_optimization4(world, features)
        assert count == len(points)
        np.testing.assert_allclose(rotation, np.eye(3), atol=1e-5)
        np.testing.assert_allclose(recovered, translation * scale, rtol=1e-4, atol=1e-5)
        for i in world:
            np.testing.assert_array_equal(world[i], snapshot[i])


def test_nonfinite_ba_solution_retains_input_geometry():
    closer, current, reference, _ = make_recovery_scene()
    camera = closer.optimizer.mK
    optimizer = Optimizer(camera, 100, 100)
    translation = np.array([0.3, 0., 0.1])
    current.pose[:3, 3] = translation + np.array([0.02, 0.01, 0.01])
    for key, point in reference.points.items():
        point.erase_observation(2)
        camera_point = point.get_world_pos().reshape(3) + translation
        projected = camera @ camera_point
        current.update_map_point(point, projected[:2] / projected[2], key)
        point.add_observation(current.mnId)
    original_pose = current.get_pose()
    optimizer.isometry3d_to_numpy = lambda pose: np.full((4, 4), np.nan)
    poses, points = optimizer.bundle_adjustment(
        {reference.mnId: reference, current.mnId: current},
        reference.points, set_fix_id=reference.mnId,
    )
    for key, point in reference.points.items():
        np.testing.assert_array_equal(points[key], point.get_world_pos().reshape(3))
    np.testing.assert_array_equal(poses[reference.mnId], reference.get_pose())
    np.testing.assert_array_equal(poses[current.mnId], original_pose)


def test_local_ba_keeps_out_of_window_history_points_fixed():
    closer, current, reference, _ = make_recovery_scene()
    optimizer = Optimizer(closer.optimizer.mK, 100, 100)
    for key, point in reference.points.items():
        current.update_map_point(point, reference.features[key], key)
        point.add_observation(current.mnId)
    history_id = min(reference.points)
    reference.delete_map_point_match(history_id)
    history_point = current.points[history_id]
    history_point.local_recovery_protected = True
    current.features[history_id] += np.array([2.0, 1.0])
    history_point.erase_observation(reference.mnId)
    before = {key: point.get_world_pos().reshape(3) for key, point in current.points.items()}
    _, optimized = optimizer.bundle_adjustment(
        {reference.mnId: reference, current.mnId: current},
        current.points, set_fix_id=reference.mnId,
    )
    np.testing.assert_array_equal(optimized[history_id], before[history_id])
