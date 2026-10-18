# Public Release

## Branches And Launchers

`main` retains the development launchers, including `run_exvivo.sh`,
`run_unity.sh`, `run_full_slam.sh` and `run_unity_slam.sh`.
`public-release` and public exports exclude these four scripts. Use the generic
commands in [README](../README.md); `run_docker.sh` forwards a supplied Python
command and `run_slam.sh` also supports the existing configuration presets.

Install the package in your active environment before local execution. Data,
logs and machine-specific overrides in `scripts/local/` and `config/local/`
are not included. Development launcher defaults may require a local dataset
layout; override their data paths when using another machine.

## Export Without History

```bash
python3 scripts/export_public.py new_public_export
```

The export copies current source files, including uncommitted changes, without
a `.git` directory. It refuses to overwrite an existing destination. The four
development launchers are excluded even when tracked on `main`.

Review files and binary assets before publishing. This is not a comprehensive
secret scanner. Retain third-party license and author attribution. Use a public
or GitHub no-reply commit email when creating a new repository.

Removing files from a branch does not remove them or author information from
earlier commits. Publishing `public-release` does not scrub `main` or make the
repository public. Do not force-push existing history without separate review.
