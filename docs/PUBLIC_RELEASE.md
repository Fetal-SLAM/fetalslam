# Public Release

## Portable Launchers

Activate a Python environment with the required native bindings, then run:

```bash
bash scripts/run_unity.sh
bash scripts/run_unity.sh data/standard_data/Images_unity_2 --max-frames 120
bash scripts/run_exvivo.sh --max-frames 120
```

The launchers locate the repository from their own location, not the caller's
working directory. Dataset paths are relative to the repository root; absolute
paths can also be supplied locally. No local interpreter path is embedded.
Set `PYTHON_BIN` to override `python3` from the active environment.
Datasets are not distributed. Place Unity images in
`data/standard_data/Images_unity_1/images` (or `Images_unity_2/images`) and ex-vivo
images in `data/standard_data/ex_vivo_data/undist_image`.

The previous `run_full_slam.sh` and `run_unity_slam.sh` scripts, local overrides,
logs and datasets are excluded from Git, Docker builds and public exports.
Their local copies can still be used for private testing.

## Export Without History

Deleting private paths in a new commit does not remove them from older commits.
Existing commit authors and emails also remain in Git history. Do not publish
the existing history when a history-free release is required.

```bash
python3 scripts/export_public.py
cd public_release
git init
git symbolic-ref HEAD refs/heads/main
git add .
git diff --cached --stat
```

Review the files before committing. Configure the new repository's author name
and a suitable public or GitHub-provided no-reply email before the first commit.
Then commit and push to a new empty remote. Do not force-push over an existing
remote's history without a separate review and backup.

The export copies current source files, including uncommitted changes, and
contains no `.git` directory. It refuses to overwrite an existing destination.
This is a publication convenience, not a comprehensive secret scanner. Review
new source files and binary assets before publishing. Third-party license and
author attribution must remain intact.
