import click

from pm_icecon.bt.cli import cli as bt_cli
from pm_icecon.cdr import cli as cdr_cli
from pm_icecon.nt.cli import cli as nt_cli


@click.group()
def cli():
    """Run the nasateam or bootstrap algorithm."""
    ...


cli.add_command(bt_cli)  # type: ignore
cli.add_command(nt_cli)  # type: ignore
cli.add_command(cdr_cli)  # type: ignore


if __name__ == '__main__':
    from pm_icecon.cli.entrypoint import cli

    cli()  # type: ignore
