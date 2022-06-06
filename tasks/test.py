from invoke import task

from .util import PROJECT_DIR, print_and_run


@task(aliases=['flake8'])
def lint(ctx):
    """Run flake8 linting."""
    print_and_run(f'flake8 {PROJECT_DIR}')


@task(aliases=['mypy'])
def typecheck(ctx):
    """Run mypy typechecking."""
    mypy_cfg_path = PROJECT_DIR / '.mypy.ini'
    print_and_run(
        f'mypy --config-file={mypy_cfg_path} {PROJECT_DIR}/',
        pty=True,
    )

    print('🎉🦆 Type checking passed.')


@task()
def unit(ctx):
    """Run unit tests."""
    print_and_run(
        f'pytest {PROJECT_DIR}/bt/test.py',
        pty=True,
    )


@task()
def vulture(ctx):
    """Use `vulture` to detect dead code."""
    print_and_run(
        f'vulture'
        f' --exclude {PROJECT_DIR}/tasks,{PROJECT_DIR}/bt/_types.py'
        f' {PROJECT_DIR}',
        pty=True,
    )


@task(
    pre=[
        lint,
        typecheck,
        vulture,
        unit,
    ],
    default=True,
)
def all(ctx):
    """Run all of the tests."""
    ...
