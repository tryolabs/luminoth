import click

from .trainer import train
from .voc import voc
from .evaluator import evaluate
from .test import test


@click.group()
def cli():
    pass


cli.add_command(train)
cli.add_command(voc)
cli.add_command(evaluate)
cli.add_command(test)
