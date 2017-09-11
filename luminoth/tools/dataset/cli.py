import click

from .pascalvoc import pascalvoc
from .imagenet import imagenet


@click.group(help='Groups of commands to manage datasets')
def dataset():
    pass


dataset.add_command(pascalvoc)
dataset.add_command(imagenet)
