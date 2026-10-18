from pathlib import Path

import fetal_slam


def test_source_files_are_directly_under_src():
    root = Path(__file__).resolve().parents[1]
    assert (root / 'src/slam.py').is_file()
    assert (root / 'src/__init__.py').is_file()
    assert (root / 'src/features/modules/xfeat.py').is_file()
    assert not (root / 'src/fetal_slam').exists()


def test_feature_weights_remain_relative_to_package():
    package_dir = Path(fetal_slam.__file__).parent
    for name in ('xfeat.pt', 'xfeat-lighterglue.pt'):
        assert (package_dir / 'features/weights' / name).is_file()


def test_runtime_package_does_not_contain_legacy_modules():
    package_dir = Path(fetal_slam.__file__).parent

    assert not (package_dir / "localmapping.py").exists()
    assert not (package_dir / "loopclosing.py").exists()
    assert not (package_dir / "localmapping_thread.py").exists()
    assert not (package_dir / "loopclosing_thread.py").exists()
    assert (package_dir / "local_mapping.py").is_file()
    assert (package_dir / "loop_closing.py").is_file()
