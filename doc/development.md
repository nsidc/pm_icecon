# Developing `pm_icecon`

## Adding dependencies

To add new dependencies to this project, update the `environment.yml` file with
the new dependency. Then update your conda environment:

```
$ mamba env update
```

Once the conda environment has been updated, lock the environment using `conda-lock`:

```
$ conda-lock -p linux-64
```

Commit the changes for the `environment.yml` and the `conda-lock.yml` files.



## Linting / formatting
This project uses [pre-commit](https://pre-commit.com/) to run pre-commit hooks
that check and format this project's code for stylistic consistency (using
`ruff` and `black`) .

The pre-commit configuration for this project can be found in
`.pre-commit-config.yaml`. Configuration for specific tools (e.g., `vulture`) is
given in the included `pyproject.toml`.

For more information about using `pre-commit`, please sese the [Scientific
Python Library Development Guide's section on
pre-commit](https://learn.scientific-python.org/development/guides/gha-basic/#pre-commit).

To install pre-commit to run checks for each commit you make:

```
$ pre-commit install
```

To manually run the pre-commit hooks without a commit:

```
$ pre-commit run --all-files
```

## Invoke (tests/common tasks)

This project uses `invoke` as a task runner. To see all of the available tasks:

```
$ invoke -l
Available tasks:

  test.all (test)              Run all of the tests.
  test.ci                      Run tests not requiring access to external data.
  test.regression              Run regression tests.
  test.typecheck (test.mypy)   Run mypy typechecking.
  test.unit                    Run unit tests.
```


## Making a new release

To make a new release:

1. Make a feature branch and commit your changes.
2. Update the CHANGELOG.md and bump the version (`bumpversion {major|minor|patch}`)
3. Open a PR and have it reviewed.
4. If tests pass and the PR has been approved, merge with `main`.
5. Tag the last commit in `main` with the version you bumped to. This will
   automatically trigger a GitHub Actions build that includes deployment of the
   pacakge to anaconda.org.
