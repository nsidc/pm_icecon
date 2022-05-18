from invoke import task

from .util import PROJECT_DIR, print_and_run


@task(aliases=['flake8'])
def lint(ctx):
    """Run flake8 linting."""
    print_and_run(f'flake8 {PROJECT_DIR}')


@task(aliases=['mypy'])
def typecheck(ctx):
    """Run mypy typechecking."""
    print_and_run(
        f'mypy --config-file=.mypy.ini {PROJECT_DIR}/',
        pty=True,
    )

    print('🎉🦆 Type checking passed.')


@task(pre=[
    lint,
    typecheck,
], default=True)
def all(ctx):
    """Run all of the tests."""
    ...
