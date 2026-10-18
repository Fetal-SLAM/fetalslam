from pathlib import Path

import fetal_slam


def test_runtime_package_does_not_contain_legacy_modules():
    package_dir = Path(fetal_slam.__file__).parent

    assert not (package_dir / "localmapping.py").exists()
    assert not (package_dir / "loopclosing.py").exists()
    assert not (package_dir / "localmapping_thread.py").exists()
    assert not (package_dir / "loopclosing_thread.py").exists()
    assert (package_dir / "local_mapping.py").is_file()
    assert (package_dir / "loop_closing.py").is_file()
