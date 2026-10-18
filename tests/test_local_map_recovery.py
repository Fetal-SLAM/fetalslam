from types import SimpleNamespace

import numpy as np

from fetal_slam.loop_closing import LoopClosing


class RecoveryPoint:
    def __init__(self, point_id, position, observations):
        self.mnId = point_id
        self.position = np.asarray(position, dtype=float).reshape(3, 1)
        self.observations = dict(observations)

    def get_world_pos(self):
        return self.position.copy()

    def set_world_pos(self, position):
        self.position = np.asarray(position, dtype=float).reshape(3, 1)

    def get_observations(self):
        return self.observations.copy()

    def add_observation(self, keyframe_id):
        self.observations[keyframe_id] = True

    def erase_observation(self, keyframe_id):
        self.observations.pop(keyframe_id, None)

    def Observations(self):
        return len(self.observations)

    def is_bad(self):
        return False


class RecoveryKeyFrame:
    def __init__(self, keyframe_id, frame_id, image, pose=None):
        self.mnId = keyframe_id
        self.mnFrameId = frame_id
        self.image = image
        self.pose = np.eye(4) if pose is None else pose.copy()
        self.features = {}
        self.points = {}

    def get_pose(self):
        return self.pose.copy()

    def set_pose(self, pose):
        self.pose = np.asarray(pose, dtype=float).copy()

    def get_xfeat_features(self):
        return {"keyframe": self.mnId}

    def get_KeysUn_dict(self):
        return {key: value.copy() for key, value in self.features.items()}

    def get_KeysUn_dict_feature(self, point_id):
        feature = self.features.get(point_id)
        return None if feature is None else feature.copy()

    def get_mvpMapPoints(self):
        return self.points.copy()

    def update_map_point(self, point, feature, point_id):
        self.points[point_id] = point
        self.features[point_id] = np.asarray(feature, dtype=float).copy()
        return True

    def delete_map_point_match(self, point_id):
        self.points.pop(point_id, None)
        self.features.pop(point_id, None)

    def compute_scene_median_depth(self, quantile):
        return 4.0


class IdentityMatcher:
    def match(self, reference, current):
        return True, np.eye(3)

    def warp_shitomasi_features(self, features, homography):
        return np.asarray(features, dtype=float).reshape(-1, 1, 2)


class FullOverlap:
    def compute_iou_from_homography_fast(self, homography):
        return 1.0


class RecordingOptimizer:
    def __init__(self, camera_matrix):
        self.mK = camera_matrix
        self.calls = []

    def bundle_adjustment(self, frames, points, set_fix_id=0):
        self.calls.append((set(frames), set(points), set_fix_id))
        return (
            {frame_id: frame.get_pose() for frame_id, frame in frames.items()},
            {point_id: point.get_world_pos() for point_id, point in points.items()},
        )


def make_recovery_scene():
    width = height = 100
    camera_matrix = np.array(
        [[80.0, 0.0, 50.0], [0.0, 80.0, 50.0], [0.0, 0.0, 1.0]]
    )
    image = np.random.default_rng(8).integers(
        0, 256, (height, width, 3), dtype=np.uint8
    )
    reference = RecoveryKeyFrame(4, 10, image)
    current = RecoveryKeyFrame(20, 100, image.copy())

    point_id = 1000
    for row, y in enumerate(np.linspace(-0.8, 0.8, 4)):
        for column, x in enumerate(np.linspace(-1.0, 1.0, 6)):
            z = 3.5 + 0.15 * ((row + column) % 4)
            position = np.array([x, y, z])
            feature = np.array(
                [80.0 * x / z + 50.0, 80.0 * y / z + 50.0]
            )
            point = RecoveryPoint(
                point_id, position, {reference.mnId: True, 2: True}
            )
            reference.points[point_id] = point
            reference.features[point_id] = feature
            point_id += 1

    tree = SimpleNamespace(
        parents={current.mnId: {"node": SimpleNamespace(get_kf=lambda: current)}},
        grand_parents={
            reference.mnId: SimpleNamespace(get_kf=lambda: reference)
        },
    )
    optimizer = RecordingOptimizer(camera_matrix)
    closer = LoopClosing(width=width, height=height)
    closer.set_KeyFrameTree(tree)
    closer.set_xfeat(IdentityMatcher())
    closer.set_vis_recon(FullOverlap())
    closer.set_optimizer(optimizer)
    closer.configure_local_map_recovery(
        enabled=True,
        interval=1,
        min_frame_gap=30,
        candidate_count=2,
        iou_threshold=0.3,
        min_inliers=8,
        min_observations=2,
        max_points=50,
        reprojection_error=2.0,
        max_pose_jump_depth_factor=0.1,
    )

    return closer, current, reference, optimizer


def test_background_recovery_adds_only_geometrically_verified_old_points():
    closer, current, reference, optimizer = make_recovery_scene()
    original_pose = current.get_pose()
    original_points = {key: point.get_world_pos() for key, point in reference.points.items()}
    result = closer.recover_local_map(current.mnId)

    assert result["accepted"]
    assert result["candidate_id"] == reference.mnId
    assert result["inliers"] >= 8
    assert result["added_points"] == result["inliers"]
    assert len(current.points) == result["added_points"]
    assert all(current.mnId in point.observations for point in current.points.values())
    assert optimizer.calls == []
    np.testing.assert_allclose(current.get_pose(), original_pose, atol=1e-6)
    for key, point in reference.points.items():
        np.testing.assert_array_equal(point.get_world_pos(), original_points[key])


def test_repeated_recovery_preserves_scale_and_shared_history():
    closer, current, reference, optimizer = make_recovery_scene()
    positions = {key: point.get_world_pos() for key, point in reference.points.items()}
    for cycle in range(20):
        current.mnId = 100 + cycle
        current.features.clear()
        current.points.clear()
        closer.KFtree.parents = {current.mnId: {"node": SimpleNamespace(get_kf=lambda: current)}}
        result = closer.recover_local_map(current.mnId)
        assert result['accepted']
        np.testing.assert_allclose(current.get_pose(), np.eye(4), atol=1e-6)
        for key, point in reference.points.items():
            np.testing.assert_array_equal(point.get_world_pos(), positions[key])
            assert reference.mnId in point.observations
    assert optimizer.calls == []


def test_pose_inconsistent_history_is_rejected_even_when_pnp_fits():
    closer, current, reference, optimizer = make_recovery_scene()
    current.pose[0, 3] = 1.0
    original_pose = current.get_pose()
    result = closer.recover_local_map(current.mnId)
    assert not result['accepted']
    assert not current.points
    np.testing.assert_array_equal(current.get_pose(), original_pose)
    assert optimizer.calls == []


def test_small_pose_drift_is_corrected_without_moving_map_points():
    closer, current, reference, optimizer = make_recovery_scene()
    current.pose[0, 3] = 0.02
    points = {key: point.get_world_pos() for key, point in reference.points.items()}
    result = closer.recover_local_map(current.mnId)
    assert result['accepted']
    np.testing.assert_allclose(current.get_pose(), np.eye(4), atol=1e-6)
    for key, point in reference.points.items():
        np.testing.assert_array_equal(point.get_world_pos(), points[key])
    assert optimizer.calls == []


def test_projection_gate_rejects_behind_camera_and_duplicate_observations():
    closer, current, reference, _ = make_recovery_scene()
    point_id = next(iter(reference.points))
    point = reference.points[point_id]
    feature = reference.features[point_id]
    matches = [(point_id, point, feature), (point_id + 100, point, feature)]
    assert closer._filter_recovery_observations(current, matches, [0, 1]).tolist() == [0]
    point.set_world_pos(-point.get_world_pos())
    assert not len(closer._filter_recovery_observations(current, matches, [0, 1]))


def test_failed_insertion_rolls_back_observations():
    closer, current, reference, _ = make_recovery_scene()
    original_update = current.update_map_point
    def fail_after_first(point, feature, point_id):
        if current.points:
            raise RuntimeError('injected insertion failure')
        return original_update(point, feature, point_id)
    current.update_map_point = fail_after_first
    result = closer.recover_local_map(current.mnId)
    assert not result['accepted']
    assert result['reason'] == 'insertion-failed'
    assert not current.points and not current.features
    assert all(current.mnId not in point.observations for point in reference.points.values())


def test_accepted_recovery_refreshes_foreground_tracking_reference():
    from fetal_slam.slam import XFeatSlamOffline

    slam = XFeatSlamOffline.__new__(XFeatSlamOffline)
    slam.local_mapping_worker = SimpleNamespace(poll=lambda: [], busy=False)
    result = SimpleNamespace(error=None, value={
        'recovery': {'accepted': True, 'added_points': 20, 'candidate_id': 4, 'inliers': 20},
        'loop': (None, None, None),
    })
    slam.loop_closing_worker = SimpleNamespace(poll=lambda: [result], busy=False)
    slam.local_map_recovery_count = 0
    slam.local_map_recovered_points = 0
    slam.kf_tree = SimpleNamespace(get_all_keyframes_for_loopclosing=lambda: {})
    calls = []
    def adopt(track_current):
        calls.append(track_current)
        return 'refreshed'
    slam.adopt_new_parent = adopt
    assert slam.consume_background_results(track_current=False) == 'refreshed'
    assert calls == [False]
    assert slam.local_map_recovery_count == 1
    assert slam.local_map_recovered_points == 20
