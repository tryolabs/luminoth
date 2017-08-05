import click

from .gcloud import gc


@click.group(help='Groups of commands to train models in the cloud')
def cloud():
    pass


cloud.add_command(gc)
