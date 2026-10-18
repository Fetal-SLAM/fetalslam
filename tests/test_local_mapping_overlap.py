from types import SimpleNamespace

import numpy as np

from fetal_slam.local_mapping import LocalMapping
from test_local_map_recovery import make_recovery_scene


def make_scene():
    closer, current, reference, optimizer = make_recovery_scene()
    mapping = LocalMapping(100, 100)
    mapping.set_optimizer(optimizer)
    mapping.set_KeyFrameTree(SimpleNamespace(
        parents={reference.mnId: {"node": SimpleNamespace(get_kf=lambda: reference)}},
    ))
    return mapping, current, reference


def test_overlap_reuses_ids_and_preserves_geometry():
    mapping, current, reference = make_scene()
    before = {key: point.get_world_pos() for key, point in reference.points.items()}
    pose = current.get_pose()
    count = mapping.associate_existing_points(current)
    assert count == len(reference.points)
    for key, point in current.points.items():
        assert point is reference.points[key]
        assert current.mnId in point.get_observations()
        np.testing.assert_array_equal(point.get_world_pos(), before[key])
    np.testing.assert_array_equal(current.get_pose(), pose)
    assert mapping.associate_existing_points(current) == 0


def test_overlap_rejects_inconsistent_geometry_and_noisy_images():
    mapping, current, reference = make_scene()
    for point in reference.points.values():
        point.position[0] += 1.0
    assert mapping.associate_existing_points(current) == 0
    mapping, current, reference = make_scene()
    current.image = np.zeros_like(current.image)
    assert mapping.associate_existing_points(current) == 0


def test_overlap_budget_and_duplicate_pixels():
    mapping, current, reference = make_scene()
    mapping.max_fake_features = 5
    assert mapping.associate_existing_points(current) == 5
    assert mapping.associate_existing_points(current) == 0
    mapping, current, reference = make_scene()
    first = next(iter(reference.points))
    current.features[-1] = reference.features[first].copy()
    assert mapping.associate_existing_points(current) == len(reference.points) - 1
    assert first not in current.points


def test_invisible_dictionary_entries_do_not_consume_active_budget():
    mapping, current, reference = make_scene()
    mapping.max_fake_features = 5
    current.features = {-key - 1: np.array([-100.0, -100.0]) for key in range(200)}
    current.mvTrackedPoints = np.zeros((200, 1), dtype=bool)
    assert mapping.associate_existing_points(current) == 5


def test_overlap_only_promotes_without_triangulation_or_geometry_update():
    mapping, current, reference = make_scene()
    mapping.local_point_reuse_enabled = True
    tree = mapping.KFtree
    tree.newest_parent_name = reference.mnId
    tree.new_parent_flag = False
    tree.parents[reference.mnId]["candidate"] = SimpleNamespace(kf=current)
    promotions = []
    tree.update_keyframe_tree = lambda flag, name: promotions.append((flag, name))
    tree.update_node = lambda: None
    mapping.check_new_key_frame = lambda: (True, "latest", "candidate", np.eye(3), {})
    mapping.assign_new_features_and_reconstruct = lambda *args: (False, None)
    result = mapping.run()
    assert result["reused_observations"] == len(reference.points)
    assert promotions == [(True, "candidate")]


def test_disabled_reuse_skips_association_but_keeps_normal_local_ba():
    mapping, current, reference = make_scene()
    assert not mapping.local_point_reuse_enabled
    tree = mapping.KFtree
    tree.newest_parent_name = reference.mnId
    tree.new_parent_flag = False
    calls = []
    def forbidden(*args):
        raise AssertionError("disabled local association was executed")
    mapping.associate_existing_points = forbidden
    mapping.check_new_key_frame = lambda: (True, "latest", "candidate", np.eye(3), {})
    mapping.assign_new_features_and_reconstruct = lambda *args: (True, {})
    mapping.update_map_points_to_other_child = lambda *args: calls.append("triangulation")
    mapping.optimizer.local_bundle_adjustment = lambda *args: (calls.append("BA") or ({}, {}))
    tree.update_all_keyframe_pos = lambda *args: calls.append("poses")
    mapping.map = SimpleNamespace(update_map_points_world_pos=lambda *args: calls.append("points"))
    tree.update_keyframe_tree = lambda flag, name: calls.append((flag, name))
    tree.update_node = lambda: None
    result = mapping.run()
    assert result["reused_observations"] == 0
    assert mapping.reused_feature_ids == set()
    assert calls == ["triangulation", "BA", "poses", "points", (True, "candidate")]


def test_disabled_history_recovery_does_not_touch_geometry():
    closer, current, reference, optimizer = make_recovery_scene()
    closer.local_map_recovery_enabled = False
    pose = current.get_pose()
    result = closer.recover_local_map(current.mnId)
    assert not result["attempted"]
    assert not result["accepted"]
    assert current.points == {}
    assert optimizer.calls == []
    np.testing.assert_array_equal(current.get_pose(), pose)
