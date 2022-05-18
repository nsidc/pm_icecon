from invoke import task

from .util import PROJECT_DIR, print_and_run


@task(aliases=['flake8'])
def lint(ctx):
    """Run flake8 linting."""
    print_and_run(f'flake8 {PROJECT_DIR}')
