import click

from cdr_amsr2.bt.cli import cli as bt_cli
from cdr_amsr2.nt.cli import cli as nt_cli


@click.group()
def cli():
    """Run the nasateam or bootstrap algorithm."""
    ...


cli.add_command(bt_cli)
cli.add_command(nt_cli)


if __name__ == '__main__':
    cli()
