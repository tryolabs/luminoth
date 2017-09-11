import click

from .transform import transform


@click.group(help='Groups of commands to manage datasets')
def dataset():
    pass


dataset.add_command(transform)
