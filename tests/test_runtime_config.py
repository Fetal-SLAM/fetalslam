import ast
import logging
from pathlib import Path
import tempfile

import fetal_slam
from fetal_slam.parameters import Parameter
from fetal_slam.runtime_config import runtime_settings, configure_logging
from fetal_slam.optimizer import Optimizer
from fetal_slam.local_mapping import LocalMapping
import numpy as np
import threading
from types import SimpleNamespace
import torch
from fetal_slam.features.detector import Xfeat_Det_Matching


def test_runtime_defaults_and_overrides_are_independent():
    first = runtime_settings('optimizer', {'iterations': 25})
    assert first.iterations == 25
    assert runtime_settings('optimizer').iterations == 10
    optimizer = Optimizer(np.eye(3), 720, 720, settings=first)
    assert optimizer.settings.iterations == 25
    mapper = LocalMapping(720, 720, settings=runtime_settings('local_mapping', {'reuse_max_points': 17}))
    assert mapper.settings.reuse_max_points == 17


def test_runtime_config_rejects_invalid_values():
    for section, values in (
        ('optimizer', {'iterations': 0}),
        ('optimizer', {'iterations': True}),
        ('optimizer', {'mono_chi2': float('nan')}),
        ('optimizer', {'unknown': 1}),
        ('features', {'lightglue_min_confidence': 1.1}),
        ('features', {'ransac_confidence': 1.0}),
        ('features', {'corner_block_size': 4}),
        ('features', {'homography_min_matches': 3}),
        ('tracking', {'image_margin': 0.5}),
        ('viewer', {'near_clip': 1001}),
        ('logging', {'level': 'VERBOSE'}),
        ('local_mapping', []),
    ):
        try:
            runtime_settings(section, values)
        except ValueError:
            continue
        raise AssertionError((section, values))
    assert runtime_settings('system_config', {'loop_check_interval': 0}).loop_check_interval == 0


def test_presets_load_with_shared_runtime_defaults():
    root = Path(__file__).resolve().parents[1]
    for profile, camera in [('vo_fast.yaml', 'feto_cam_exvivo.yaml'), ('vo_unity.yaml', 'feto_cam.yaml')]:
        params = Parameter()
        params.load_sys_param(root/'config/param'/profile, root/'config/param'/camera)
        assert params.tracker.max_kpts == 3000
        assert params.runtime_config['tracking'].max_parents == 30
        assert params.dense_config['surface_mode'] == 'plane_grid'


def test_logging_is_idempotent_and_keeps_root_handlers():
    root_handlers = list(logging.getLogger().handlers)
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder)/'slam.log'
        settings = runtime_settings('logging', {'level': 'WARNING', 'file': str(path)})
        configure_logging(settings)
        configure_logging(settings)
        logger = logging.getLogger('fetal_slam.test')
        logger.debug('hidden')
        logger.warning('visible')
        text = path.read_text()
        assert 'hidden' not in text
        assert text.count('visible') == 1
        assert logging.getLogger().handlers == root_handlers
        configure_logging(runtime_settings('logging'))


def test_runtime_contains_no_print_calls_or_retired_scripts():
    package = Path(fetal_slam.__file__).parent
    for path in package.rglob('*.py'):
        tree = ast.parse(path.read_text())
        assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'print' for n in ast.walk(tree)), path
    assert not (Path(__file__).resolve().parents[1]/'extras').exists()


def test_no_silently_overwritten_class_methods():
    package = Path(fetal_slam.__file__).parent
    for path in package.rglob('*.py'):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ClassDef):
                names = [method.name for method in node.body if isinstance(method, ast.FunctionDef)]
                assert len(names) == len(set(names)), (path, node.name)


def test_initial_matching_keeps_current_frame_after_demo_removal():
    detector = object.__new__(Xfeat_Det_Matching)
    detector.inference_lock = threading.RLock()
    detector.ref_frame = np.zeros((8, 8, 3), dtype=np.uint8)
    features = {'keypoints': torch.zeros((5, 2)), 'descriptors': torch.zeros((5, 64)), 'image_size': (8, 8)}
    detector.ref_precomp = features
    detector.lightglue_min_confidence = 0.1
    descriptor = SimpleNamespace(detectAndCompute=lambda image: features)
    matcher = SimpleNamespace(match_lighterglue=lambda *args, **kwargs: (torch.arange(5), torch.arange(5)))
    detector.method = SimpleNamespace(descriptor=descriptor, matcher=matcher)
    image = np.ones((8, 8, 3), dtype=np.uint8)
    matched, homography, pairs = detector.detect_and_match(image)
    assert not matched and homography is None and pairs is None
    assert np.array_equal(detector.current_frame, image)
    assert not np.shares_memory(detector.current_frame, image)
