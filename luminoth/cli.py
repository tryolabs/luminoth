"""Simple command line utility called `lumi`.

The cli is composed of subcommands that are able to handle different tasks
needed for training and using deep learning models.

It's base subcommands are:
    train: For training locally.
    cloud: For traning and monitoring in the cloud.
    dataset: For modifying and transforming datasets.
"""

import click

from luminoth.eval import eval
from luminoth.predict import predict
from luminoth.tools import cloud, dataset, server
from luminoth.train import train


CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.group(context_settings=CONTEXT_SETTINGS)
def cli():
    pass


cli.add_command(cloud)
cli.add_command(dataset)
cli.add_command(predict)
cli.add_command(eval)
cli.add_command(train)
cli.add_command(server)
