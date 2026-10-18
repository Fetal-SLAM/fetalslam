import threading
from types import SimpleNamespace

import torch

from fetal_slam.features.detector import Xfeat_Det_Matching


class FewMatchBackend:
    def match_lighterglue(self, features1, features2, min_conf):
        assert features1['image_size'] == (640, 480)
        assert features2['image_size'] == (640, 480)
        assert min_conf == 0.1
        indices = torch.arange(5)
        return indices, indices


def test_match_rejects_too_few_correspondences_without_unbound_result():
    detector = object.__new__(Xfeat_Det_Matching)
    detector.inference_lock = threading.RLock()
    detector.lightglue_min_confidence = 0.1
    detector.ransac_thr = 4.0
    detector.min_inliers = 50
    detector.method = SimpleNamespace(matcher=FewMatchBackend())

    precomputed = {
        "image_size": (640, 480),
        "keypoints": torch.zeros((5, 2)),
        "descriptors": torch.zeros((5, 64)),
    }

    matched, homography = detector.match(precomputed, precomputed)

    assert not matched
    assert homography is None


def test_lightglue_rejects_missing_image_size_without_mnn_fallback():
    detector = object.__new__(Xfeat_Det_Matching)
    features = {'keypoints': torch.zeros((5, 2)), 'descriptors': torch.zeros((5, 64))}
    try:
        detector.matched_points(features, features)
    except ValueError as error:
        assert 'image_size' in str(error)
    else:
        raise AssertionError('Missing image size must not silently use another matcher')


def test_descriptor_wrapper_records_real_image_dimensions():
    import numpy as np
    from fetal_slam.features.detector import CVWrapper
    backend = SimpleNamespace(detectAndCompute=lambda image: [{'keypoints': torch.zeros((0, 2))}])
    features = CVWrapper(backend).detectAndCompute(np.zeros((120, 240, 3), dtype=np.uint8))
    assert features['image_size'] == (240, 120)


def test_lightglue_empty_features_do_not_invoke_network():
    detector = object.__new__(Xfeat_Det_Matching)
    empty = {'keypoints': torch.zeros((0, 2)), 'descriptors': torch.zeros((0, 64))}
    first, second = detector.matched_points(empty, empty)
    assert first.shape == second.shape == (0, 2)


def test_bundled_lightglue_checkpoint_loads_without_global_config_mutation():
    from kornia.feature.lightglue import LightGlue
    from fetal_slam.features.modules.lighterglue import LighterGlue
    original = dict(LightGlue.default_conf)
    model = LighterGlue()
    assert LightGlue.default_conf == original
    assert not model.net.training
    assert model.net.conf.input_dim == 64
    assert model.net.conf.n_layers == 6
