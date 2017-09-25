import click

from .web import web


@click.group(help='Groups of commands to serve models')
def server():
    pass


server.add_command(web)
