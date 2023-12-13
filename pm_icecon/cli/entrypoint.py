import click

from pm_icecon.nt.cli import cli as nt_cli


@click.group()
def cli():
    """Run the nasateam or bootstrap algorithm."""
    ...


cli.add_command(nt_cli)


if __name__ == "__main__":
    from pm_icecon.cli.entrypoint import cli

    cli()
