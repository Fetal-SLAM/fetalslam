import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]


def test_public_launchers_resolve_paths_from_another_directory():
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)/'project with spaces'
        scripts = root/'scripts'
        scripts.mkdir(parents=True)
        for name in ('run_slam.sh',):
            shutil.copy2(ROOT/'scripts'/name, scripts/name)
        for name in ('Images_unity_1', 'Images_unity_2', 'ex_vivo_data'):
            (root/'data/standard_data'/name).mkdir(parents=True)
        python = root/'fake python'
        python.write_text('#!' + sys.executable + '\nimport json,os,sys\nprint(json.dumps([os.getcwd(),sys.argv[1:],os.environ.get("PYTHONPATH")]))\n')
        python.chmod(0o755)
        env = dict(os.environ, PYTHON_BIN=str(python))
        for profile, data, camera, subdir in (
            ('unity', 'Images_unity_2', 'feto_cam.yaml', 'images'),
            ('exvivo', 'ex_vivo_data', 'feto_cam_exvivo.yaml', 'undist_image'),
        ):
            dataset = 'data/standard_data/' + data
            output = subprocess.check_output(['bash', str(scripts/'run_slam.sh'), profile, dataset, '--max-frames', '120'], cwd=folder, env=env)
            cwd, args, pythonpath = json.loads(output)
            assert cwd == str(root)
            assert args[:2] == ['-m', 'fetal_slam']
            assert args[3] == 'config/param/' + camera
            assert args[4] == dataset
            assert args[-3:] == [subdir, '--max-frames', '120']
            assert pythonpath == env.get('PYTHONPATH')
        assert subprocess.run(['bash', str(scripts/'run_slam.sh'), 'invalid'], cwd=folder, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 2
        assert subprocess.run(['bash', str(scripts/'run_slam.sh'), 'unity', 'missing'], cwd=folder, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 2
        subprocess.check_call(['bash', str(scripts/'run_slam.sh'), '--help'], cwd=folder, stdout=subprocess.DEVNULL)


def test_public_export_excludes_private_files_and_git_history():
    export = runpy.run_path(str(ROOT/'scripts/export_public.py'))['export_public']
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)/'source'
        root.mkdir()
        subprocess.check_call(['git', 'init', '-q', str(root)])
        for name in ('scripts/run_slam.sh', 'scripts/run_unity.sh', 'scripts/run_exvivo.sh', 'scripts/run_unity_slam.sh', 'scripts/run_full_slam.sh', 'scripts/local/test.sh', 'config/local/test.yaml', 'debug.txt', '.env', 'data/private.txt', 'README.md'):
            path = root/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('sample\n')
        subprocess.check_call(['git', 'add', '.'], cwd=root)
        (root/'README.md').write_text('current uncommitted content\n')
        destination = Path(folder)/'public'
        assert export(root, destination) == 2
        assert (destination/'README.md').read_text() == 'current uncommitted content\n'
        assert sorted(str(p.relative_to(destination)) for p in destination.rglob('*') if p.is_file()) == ['README.md', 'scripts/run_slam.sh']
        assert not (destination/'.git').exists()
        try:
            export(root, destination)
        except ValueError:
            pass
        else:
            raise AssertionError('Export must not overwrite an existing directory')


def test_docker_entrypoint_checks_display_and_forwards_commands():
    env = dict(os.environ)
    env.pop('DISPLAY', None)
    entrypoint = ROOT/'docker/entrypoint.sh'
    result = subprocess.run(['bash', str(entrypoint), 'unity'], env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert result.returncode == 2 and b'DISPLAY' in result.stderr
    output = subprocess.check_output(['bash', str(entrypoint), sys.executable,
                                      '-c', 'print("forwarded")'], env=env)
    assert output.strip() == b'forwarded'
    subprocess.check_call(['bash', str(ROOT/'scripts/run_docker.sh'), '--help'],
                          env=env, stdout=subprocess.DEVNULL)
