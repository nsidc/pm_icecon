from invoke import task

from .util import PROJECT_DIR, print_and_run


@task(aliases=['flake8'])
def lint(ctx):
    """Run flake8 linting."""
    print_and_run(
        f'flake8 --exclude {PROJECT_DIR}/nt_tiepoint_generation {PROJECT_DIR}'
    )


@task(aliases=['mypy'])
def typecheck(ctx):
    """Run mypy typechecking."""
    mypy_cfg_path = PROJECT_DIR / '.mypy.ini'
    print_and_run(
        (
            f'PYTHONPATH={PROJECT_DIR} mypy --config-file={mypy_cfg_path}'
            f' {PROJECT_DIR}/'
        ),
        pty=True,
    )

    print('🎉🦆 Type checking passed.')


@task()
def unit(ctx):
    """Run unit tests."""
    print_and_run(
        f'PYTHONPATH={PROJECT_DIR} pytest -s {PROJECT_DIR}/cdr_amsr2/tests/unit',
        pty=True,
    )


@task()
def regression(ctx):
    """Run regression tests.

    Requires access to data on NFS and should be run on a VM.
    """
    print_and_run(
        f'PYTHONPATH={PROJECT_DIR} pytest -s {PROJECT_DIR}/cdr_amsr2/tests/regression',
        pty=True,
    )


@task()
def vulture(ctx):
    """Use `vulture` to detect dead code."""
    print_and_run(
        (
            'vulture'
            f' --exclude {PROJECT_DIR}/tasks,{PROJECT_DIR}/nt_tiepoint_generation'
            # ignore `_types.py` because vulture doesn't understand typed dicts.
            f',{PROJECT_DIR}/cdr_amsr2/bt/_types.py'
            # ignore `base_model.py` because vulture flags config options as
            # unused variables/class.
            f',{PROJECT_DIR}/cdr_amsr2/config/models/base_model.py'
            f' {PROJECT_DIR}'
        ),
        pty=True,
    )


@task(
    pre=[
        lint,
        typecheck,
        vulture,
        unit,
    ],
)
def ci(ctx):
    """Run tests in CircleCI.

    Excludes regression tests that require access to data on NSIDC-specific
    infrastructure.
    """
    ...


@task(
    pre=[
        lint,
        typecheck,
        vulture,
        unit,
        regression,
    ],
    default=True,
)
def all(ctx):
    """Run all of the tests."""
    ...
