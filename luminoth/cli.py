import click

from .tools import dataset
from .train import train


@click.group()
def cli():
    pass


cli.add_command(dataset)
cli.add_command(train)
