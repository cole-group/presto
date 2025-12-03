# Development

## Writing Code

To create a development environment, you must have [`pixi` installed](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html).

A development environment (the default environment) can be created and activated with:

```shell
pixi shell
```

Some handy `pixi` commands are available:
```shell
pixi run lint # Lint the codebase with Ruff
pixi run format # Format the codebase with Ruff
pixi run type-check # Type-check the codebase with Mypy
pixi run test # Run the unit tests with Pytest
```

To build and the documentation locally:

```shell
pixi run docs-build
pixi run docs-serve
```

To deploy the documentation (not usually necessary as it should be automatically deployed by GH Actions):

```shell
pixi run docs-deploy
```

## Publishing

### PyPI

NOTE: Not yet implemented.

There is a GitHub Actions workflow that will automatically publish to PyPI when a new tag is pushed:
```shell
git tag <new version>
git push origin <new version>
```
