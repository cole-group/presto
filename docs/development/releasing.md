# Releasing

## Conda-Forge

`presto` is not yet available on conda-forge.

## Versioning

To add a new tag:

```shell
git tag <new version>
git push origin <new version>
```

This will trigger a new build of the docs.

Version numbers are derived from git tags using `hatch-vcs` (configured in `pyproject.toml [tool.hatch.version]`). The `presto.__version__` string is set at install time.
