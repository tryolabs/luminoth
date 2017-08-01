import click

from .voc import voc


@click.group()
def dataset():
    pass


dataset.add_command(voc)
