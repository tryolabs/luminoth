import click

from .voc import voc


@click.group(help='Groups of commands to manage datasets')
def dataset():
    pass


dataset.add_command(voc)
