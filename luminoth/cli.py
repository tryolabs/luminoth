import click

from .train import train
from .tools.voc import voc


@click.group()
def cli():
    pass

cli.add_command(train)
