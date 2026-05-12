# Releasing

## PyPI

!!! note "Not yet implemented"
    PyPI publishing is wired up in CI but has not yet shipped a release. See `.github/workflows` for the workflow definition.

A GitHub Actions workflow will publish to PyPI when a new tag is pushed:

```shell
git tag <new version>
git push origin <new version>
```

The same tag push triggers a versioned docs deploy via `mike` (see **[Building the docs](docs.md)**).

## Versioning

Version numbers are derived from git tags using `hatch-vcs` (configured in `pyproject.toml [tool.hatch.version]`). The `presto.__version__` string is set at install time.
