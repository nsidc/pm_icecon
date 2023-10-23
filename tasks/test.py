from invoke import task

from .util import PROJECT_DIR, print_and_run


@task(aliases=["mypy"])
def typecheck(ctx):
    """Run mypy typechecking."""
    print_and_run(
        ("mypy"),
        pty=True,
    )

    print("🎉🦆 Type checking passed.")


@task()
def unit(ctx):
    """Run unit tests."""
    print_and_run(
        f"pytest -s {PROJECT_DIR}/pm_icecon/tests/unit",
        pty=True,
    )


@task()
def regression(ctx):
    """Run regression tests.

    Requires access to data on NFS and should be run on a VM.
    """
    print_and_run(
        f"pytest -s {PROJECT_DIR}/pm_icecon/tests/regression",
        pty=True,
    )


@task(
    pre=[
        typecheck,
        unit,
    ],
)
def ci(ctx):
    """Run tests not requiring access to external data.

    Excludes e.g., regression tests that require access to data on
    NSIDC-specific infrastructure.
    """
    ...


@task(
    pre=[
        typecheck,
        unit,
        regression,
    ],
    default=True,
)
def all(ctx):
    """Run all of the tests."""
    ...
