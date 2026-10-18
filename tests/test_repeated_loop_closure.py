import numpy as np
import threading
from types import SimpleNamespace

from fetal_slam.keyframe import KeyFrame, KeyFrameTree
from fetal_slam.loop_closing import LoopClosing
from fetal_slam.map import Mapping
from fetal_slam.map_point import MapPoint
from fetal_slam.optimizer import Optimizer


class FakeKeyFrame:
    def __init__(self, keyframe_id):
        self.mnId = keyframe_id
        self.points = {}
        self.features = {}

    def get_mvpMapPoints(self):
        return self.points.copy()

    def get_KeysUn_dict(self):
        return self.features.copy()

    def get_xfeat_features(self):
        return {"fake": True}

    def get_map_point(self, map_point_id):
        return self.points.get(map_point_id)

    def update_map_point(self, map_point, feature, map_point_id):
        if map_point is None:
            return False
        self.points[map_point_id] = map_point
        self.features[map_point_id] = np.asarray(feature).copy()
        return True

    def delete_map_point_match(self, map_point_id):
        point_removed = self.points.pop(map_point_id, None) is not None
        feature_removed = self.features.pop(map_point_id, None) is not None
        return point_removed or feature_removed


class IdentityFeatureMatcher:
    def match(self, reference, target):
        return True, np.eye(3)

    def warp_shitomasi_features(self, features, homography):
        return np.asarray(features, dtype=float).reshape(-1, 1, 2)


def test_redundant_point_cleanup_is_repeatable_with_sparse_ids():
    mapping = Mapping()
    closer = LoopClosing()
    closer.set_map(mapping)

    point = MapPoint(np.array([[1.0], [2.0], [3.0]]))
    point.mnId = 10007
    frame_a = FakeKeyFrame(12)
    frame_b = FakeKeyFrame(4096)
    for frame in (frame_a, frame_b):
        frame.points[point.mnId] = point
        frame.features[point.mnId] = np.array([10.0, 20.0])
        point.add_observation(frame.mnId)
    mapping.add_map_point(point, point.mnId)

    keyframes = {frame_a.mnId: frame_a, frame_b.mnId: frame_b}
    closer.remove_redundant_map_points(keyframes, {point.mnId: True})

    assert mapping.get_map_point(point.mnId) is None
    assert point.get_observations() == {}
    assert all(point.mnId not in frame.points for frame in keyframes.values())

    # A later closure may encounter an old keyframe containing a stale ID.
    frame_b.points[point.mnId] = point
    frame_b.features[point.mnId] = np.array([10.0, 20.0])
    closer.remove_redundant_map_points(keyframes, {point.mnId: True})
    assert point.mnId not in frame_b.points


def test_reused_loop_parent_becomes_newest_before_culling():
    tree = KeyFrameTree(max_parents=2)
    first = FakeKeyFrame(10)
    second = FakeKeyFrame(20)
    third = FakeKeyFrame(30)

    tree.add_parent(first)
    tree.add_parent(second)
    tree.add_parent(first)
    tree.add_parent(third)

    assert list(tree.parents) == [10, 30]
    assert tree.newest_parent_name == 30


def test_reactivating_parent_preserves_existing_children():
    tree = KeyFrameTree(max_parents=3)
    parent = FakeKeyFrame(10)
    tree.add_parent(parent)
    tree.parents[parent.mnId]["children"] = ["saved-child"]
    tree.parents[parent.mnId]["saved-child"] = object()

    tree.add_parent(parent)

    assert tree.parents[parent.mnId]["children"] == ["saved-child"]
    assert "saved-child" in tree.parents[parent.mnId]


def test_shared_map_point_id_is_not_deleted_as_redundant():
    mapping = Mapping()
    closer = LoopClosing(width=100, height=100)
    closer.set_map(mapping)
    closer.set_xfeat(IdentityFeatureMatcher())

    point = MapPoint(np.array([[1.0], [2.0], [3.0]]))
    point.mnId = 77
    reference = FakeKeyFrame(10)
    target = FakeKeyFrame(20)
    for frame in (reference, target):
        frame.points[point.mnId] = point
        frame.features[point.mnId] = np.array([20.0, 30.0])
        point.add_observation(frame.mnId)
    mapping.add_map_point(point, point.mnId)

    _, redundant = closer.update_features_before_correction(
        reference.mnId,
        target.mnId,
        {reference.mnId: reference, target.mnId: target},
    )

    assert redundant == {}
    assert mapping.get_map_point(point.mnId) is point


def test_reconcile_removes_observation_missing_from_known_keyframe():
    mapping = Mapping()
    closer = LoopClosing()
    closer.set_map(mapping)
    point = MapPoint(np.array([[1.0], [2.0], [3.0]]))
    point.mnId = 501
    frame = FakeKeyFrame(42)
    point.add_observation(frame.mnId)
    mapping.add_map_point(point, point.mnId)

    repairs = closer.reconcile_map_observations({frame.mnId: frame})

    assert repairs == 1
    assert point.get_observations() == {}


def test_optimizer_vertex_ids_do_not_depend_on_sparse_map_ids():
    frame_ids = [0, 1500, 9000]
    map_point_ids = [3, 9000, 250000]

    allocated = Optimizer.allocate_vertex_ids(frame_ids, map_point_ids)

    assert not set(allocated.values()).intersection(frame_ids)
    assert len(set(allocated.values())) == len(map_point_ids)
    assert set(allocated) == set(map_point_ids)


def test_local_bundle_adjustment_always_selects_an_available_anchor():
    assert Optimizer.select_fixed_frame_id([14, 31, 90], requested_id=0) == 14
    assert Optimizer.select_fixed_frame_id([0, 14, 31], requested_id=0) == 0
    assert Optimizer.select_fixed_frame_id([], requested_id=0) is None


def test_keyframe_map_point_mutations_keep_parallel_feature_state_aligned():
    keyframe = KeyFrame.__new__(KeyFrame)
    keyframe.mMutexFeatures = threading.Lock()
    keyframe.KeysUn_dict = {10: np.array([1.0, 2.0]), 30: np.array([3.0, 4.0])}
    keyframe.mvpMapPoints = {10: object(), 30: object()}
    keyframe.mvKeysUn = np.array([[[1.0, 2.0]], [[3.0, 4.0]]], dtype=np.float32)
    keyframe.mvbOutlier = [False, True]
    keyframe.vbTriangulated = [True, True]
    keyframe.mvTrackedPoints = np.array([[True], [False]])
    keyframe.N = 2

    keyframe.delete_map_point_match(10)
    keyframe.update_map_point(object(), np.array([5.0, 6.0]), 90)

    assert list(keyframe.KeysUn_dict) == [30, 90]
    assert keyframe.mvKeysUn.reshape(-1, 2).tolist() == [[3.0, 4.0], [5.0, 6.0]]
    assert len(keyframe.mvpMapPoints) == len(keyframe.mvbOutlier) == 2
    assert len(keyframe.vbTriangulated) == len(keyframe.mvTrackedPoints) == 2


def test_initial_map_point_id_does_not_need_to_equal_feature_index():
    keyframe = KeyFrame.__new__(KeyFrame)
    keyframe.N = 2
    keyframe.mvKeysUn = np.array([[[1.0, 2.0]], [[7.0, 8.0]]], dtype=np.float32)
    keyframe.KeysUn_dict = {}
    keyframe.mvpMapPoints = {}
    point = object()

    keyframe.add_map_point(point, idx=9001, feature_index=1)

    assert keyframe.mvpMapPoints[9001] is point
    assert keyframe.KeysUn_dict[9001].tolist() == [7.0, 8.0]


def test_compact_features_preserves_map_point_id_order():
    keyframe = KeyFrame.__new__(KeyFrame)
    keyframe.mMutexFeatures = threading.Lock()
    keyframe.N = 4
    keyframe.mvKeysUn = np.array(
        [[[1.0, 2.0]], [[3.0, 4.0]], [[5.0, 6.0]], [[7.0, 8.0]]],
        dtype=np.float32,
    )
    keyframe.KeysUn_dict = {}
    keyframe.mvpMapPoints = {}
    keyframe.mvbOutlier = [False] * 4
    keyframe.vbTriangulated = [False] * 4
    keyframe.mvTrackedPoints = np.ones((4, 1), dtype=bool)
    keyframe.mvuRight = [-1] * 4
    keyframe.mvDepth = [-1] * 4
    keyframe.valid_features = 4

    first = SimpleNamespace(mnId=20)
    second = SimpleNamespace(mnId=90)
    keyframe.add_map_point(first, idx=20, feature_index=1)
    keyframe.add_map_point(second, idx=90, feature_index=3)
    keyframe.compact_features_to_map_points()

    assert keyframe.N == 2
    assert list(keyframe.mvpMapPoints) == [20, 90]
    assert keyframe.mvKeysUn.reshape(-1, 2).tolist() == [[3.0, 4.0], [7.0, 8.0]]
    assert keyframe.vbTriangulated == [True, True]


def test_pose_observations_use_map_point_ids_instead_of_array_indices():
    frame = SimpleNamespace(
        mvpMapPoints={20: "point-20", 90: "point-90"},
        KeysUn_dict={
            20: np.array([13.0, 14.0]),
            90: np.array([27.0, 28.0]),
        },
    )

    observations = Optimizer.pose_observations(frame)

    assert observations[0][:3] == (0, 20, "point-20")
    assert observations[0][3].tolist() == [13.0, 14.0]
    assert observations[1][:3] == (1, 90, "point-90")
    assert observations[1][3].tolist() == [27.0, 28.0]
