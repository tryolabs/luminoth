import click

from .http_api import http_api


@click.group(help='Groups of commands to serve models')
def server():
    pass


server.add_command(http_api)
