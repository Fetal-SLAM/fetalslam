from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from fetal_slam.parameters import Parameter
from fetal_slam.slam import XFeatSlamOffline
from test_local_map_recovery import make_recovery_scene


def make_scene():
    closer, current, reference, _ = make_recovery_scene()
    current.mnFrameId = 1402
    current.get_translation = lambda: current.get_pose()[:3, 3]
    closer.last_close_loop_keyframe_id = 1252
    closer.KFtree.newest_parent_name = reference.mnId
    closer.KFtree.parents = {reference.mnId: {"node": SimpleNamespace(get_kf=lambda: reference)}}
    closer.KFtree.grand_parents_kdtree = SimpleNamespace(
        query_k_nearest_exclude_nearby_ids=lambda *a, **k: ([reference.mnId], [], []),
        update_all_points=lambda *a: None,
    )
    return closer, current, reference


def test_cooldown_boundary_and_two_distinct_query_frames():
    closer, query, reference = make_scene()
    query.mnFrameId = 1401
    assert not closer.check_closeloop_candidate(query)[0]
    assert closer.pending_loop_candidate_count == 0
    query.mnFrameId = 1402
    assert not closer.check_closeloop_candidate(query)[0]
    assert closer.pending_loop_candidate_count == 1
    assert not closer.check_closeloop_candidate(query)[0]
    assert closer.pending_loop_candidate_count == 1
    query.mnFrameId = 1422
    accepted, candidate, _ = closer.check_closeloop_candidate(query)
    assert accepted and candidate == reference.mnId


def test_unconfirmed_query_does_not_enter_map():
    closer, query, reference = make_scene()
    def forbidden(*args):
        raise AssertionError('unconfirmed query mutated the map')
    closer.KFtree.add_parent = forbidden
    closer.correct_loop = forbidden
    query.update_features_and_mappoints = forbidden
    assert closer.run(force=True, query_keyframe=query) == (None, None, None)
    assert list(closer.KFtree.parents) == [reference.mnId]
    assert closer.last_close_loop_keyframe_id == 1252


def test_confirmed_query_enters_map_before_global_correction():
    closer, query, reference = make_scene()
    assert closer.run(force=True, query_keyframe=query) == (None, None, None)
    calls = []
    query.mnFrameId = 1422
    query.update_features_and_mappoints = lambda: calls.append('compact')
    def add_parent(frame):
        calls.append('insert')
        closer.KFtree.parents[frame.mnId] = {'node': SimpleNamespace(kf=frame)}
        closer.KFtree.newest_parent_name = frame.mnId
    closer.KFtree.add_parent = add_parent
    def correct(candidate, homography):
        assert closer.KFtree.newest_parent_name == query.mnId
        assert candidate == reference.mnId
        calls.append('correct')
        return {query.mnId: query.get_pose()}, {}
    closer.correct_loop = correct
    poses, _, candidate = closer.run(force=True, query_keyframe=query)
    assert query.mnId in poses
    assert closer.last_close_loop_keyframe_id == 1422
    assert calls == ['compact', 'insert', 'correct']


def test_periodic_query_scheduled_without_new_parent_and_respects_busy_state():
    slam = XFeatSlamOffline.__new__(XFeatSlamOffline)
    slam.system_config = SimpleNamespace(loop_check_interval=20, loop_cooldown_frames=150)
    slam.loop_closing = SimpleNamespace(last_loop_query_frame_id=1402, last_close_loop_keyframe_id=1252)
    submitted = []
    slam.loop_closing_worker = SimpleNamespace(busy=False, submit=lambda value: submitted.append(value) or True)
    slam.local_mapping_worker = SimpleNamespace(busy=False)
    slam.cur_cam_frame = SimpleNamespace(get_frame_id=lambda: 1422)
    query = SimpleNamespace(KeysUn_dict={}, mvKeysUn=np.empty((0, 1, 2)))
    with patch('fetal_slam.slam.KeyFrame', return_value=query):
        assert slam.maybe_submit_loop_probe()
        slam.local_mapping_worker.busy = True
        assert not slam.maybe_submit_loop_probe()
        slam.local_mapping_worker.busy = False
        slam.loop_closing.last_loop_query_frame_id = 1422
        assert not slam.maybe_submit_loop_probe()
    assert submitted == [query]


def test_unity_restores_only_local_point_reuse():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    parameters = Parameter()
    parameters.load_sys_param(root / 'config/param/vo_unity.yaml', root / 'config/param/feto_cam.yaml')
    assert parameters.system_config.local_point_reuse_enabled
    assert not parameters.system_config.local_map_recovery_enabled
    assert parameters.system_config.loop_confirmation_count == 2
    import yaml
    configured = yaml.safe_load((root / 'config/param/vo_unity.yaml').read_text())
    assert parameters.system_config.loop_iou_threshold == configured['system_config']['loop_iou_threshold']
    assert parameters.system_config.loop_check_interval == 20
