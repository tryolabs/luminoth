"""Simple command line utility called `lumi`.

The cli is composed of subcommands that are able to handle different tasks
needed for training and using deep learning models.

It's base subcommands are:
    train: For training locally.
    cloud: For traning and monitoring in the cloud.
    dataset: For modifying and transforming datasets.
"""

import click

from .tools import cloud, dataset, server
from .train import train


@click.group()
def cli():
    pass


cli.add_command(cloud)
cli.add_command(dataset)
cli.add_command(train)
cli.add_command(server)
