import click
import json
import os
import shutil
import six
import tarfile
import tensorflow as tf
import uuid

from luminoth.utils.config import get_config


# TODO: Create directories if needed.
LUMINOTH_PATH = os.path.expanduser('~/.luminoth')
CHECKPOINT_INDEX = 'checkpoints.json'
CHECKPOINT_PATH = 'checkpoints'


def get_checkpoint(db, id_or_alias):
    """Returns checkpoint in `db` indicatedby `id_or_alias`."""
    for cp in db['checkpoints']:
        if cp['id'] == id_or_alias or cp['alias'] == id_or_alias:
            return cp


def get_checkpoint_path(checkpoint_id):
    path = os.path.join(LUMINOTH_PATH, CHECKPOINT_PATH, checkpoint_id)
    return path


# TODO: Move handling of db file into another module, with specification.
def read_checkpoint_db():
    """Reads the checkpoints database file from disk."""
    path = os.path.join(LUMINOTH_PATH, CHECKPOINT_INDEX)
    if not os.path.exists(path):
        return {'checkpoints': []}

    with open(path) as f:
        index = json.load(f)

    return index


def save_checkpoint_db(checkpoints):
    """Overwrites the database file in disk with `checkpoints`."""
    # TODO: Merge instead? Careful with deletions.
    path = os.path.join(LUMINOTH_PATH, CHECKPOINT_INDEX)
    with open(path, 'w') as f:
        json.dump(checkpoints, f)


@click.command(help='List available checkpoints.')
def list():
    db = read_checkpoint_db()

    if not db['checkpoints']:
        click.echo('No checkpoints available.')
        return

    template = '{:>12} | {:>7} | {:>10} | {:>40} | {:>13}'

    header = template.format('id', 'dataset', 'model', 'description', 'status')
    click.echo(header)
    click.echo('=' * len(header))

    for checkpoint in db['checkpoints']:
        line = template.format(
            checkpoint['id'],
            checkpoint['dataset']['name'],
            checkpoint['model']['name'],
            checkpoint['description'],
            checkpoint['status'],
        )
        click.echo(line)


@click.command(help='Display detailed information on checkpoint.')
@click.argument('id_or_alias')
def info(id_or_alias):
    db = read_checkpoint_db()

    checkpoint = get_checkpoint(db, id_or_alias)
    if not checkpoint:
        click.echo(
            "Checkpoint '{}' not found in index.".format(id_or_alias)
        )
        return

    click.echo('{} - {}'.format(checkpoint['id'], checkpoint['name']))
    click.echo('Description: {}'.format(checkpoint['description']))
    # TODO: Rest of the info.


@click.command(help='Create a checkpoint from a configuration file.')
@click.argument('config_files', nargs=-1)
@click.option(
    'override_params', '--override', '-o', multiple=True,
    help='Override model config params.'
)
@click.option('--alias', help="Specify the checkpoint's alias.")
def create(config_files, override_params, alias):
    click.echo('Creating checkpoint for given configuration...')
    # TODO: Validate alias and the rest of the commands.

    # Get and build the configuration file for the model.
    config = get_config(config_files, override_params=override_params)

    # Retrieve the files for the last checkpoint available.
    run_dir = os.path.join(config.train.job_dir, config.train.run_name)
    ckpt = tf.train.get_checkpoint_state(run_dir)
    if not ckpt or not ckpt.all_model_checkpoint_paths:
        click.echo("Couldn't find checkpoint in '{}'.".format(run_dir))
        return

    # TODO: Can we count on them being sorted and not do this?
    last_checkpoint = sorted([
        {'global_step': int(path.split('-')[-1]), 'file': path}
        for path in ckpt.all_model_checkpoint_paths
    ], key=lambda c: c['global_step'])[-1]['file']

    checkpoint_prefix = os.path.basename(last_checkpoint)
    checkpoint_paths = [
        os.path.join(run_dir, file)
        for file in os.listdir(run_dir)
        if file.startswith(checkpoint_prefix)
    ]

    # Find the `classes.json` file.
    classes_path = os.path.join(config.dataset.dir, 'classes.json')
    if not os.path.exists(classes_path):
        classes_path = None

    # Create an checkpoint_id to identify the checkpoint.
    checkpoint_id = str(uuid.uuid4()).replace('-', '')[:12]

    # Update the directory paths for the configuration file. Since it's going
    # to be packed into a single tar file, we set them to the current directoy.
    # TODO: Just empty them and hard-code when loading? Other way of doing?
    config.dataset.dir = '.'
    config.train.job_dir = '.'
    config.train.run_name = checkpoint_id

    # Create the directory that will contain the model.
    # TODO: Abstract into function for handling filesystem-related stuff.
    path = os.path.join(LUMINOTH_PATH, CHECKPOINT_PATH, checkpoint_id)
    os.makedirs(path, exist_ok=True)

    with open(os.path.join(path, 'config.yml'), 'w') as f:
        json.dump(config, f)

    # Add the checkpoint files.
    for checkpoint_path in checkpoint_paths:
        shutil.copy2(checkpoint_path, path)

    # Add `checkpoint` file to indicate where the checkpoint is located. We
    # need to create it manually instead of just copying as it may contain
    # absolute paths.
    with open(os.path.join(path, 'checkpoint'), 'w') as f:
        f.write(
            """
            model_checkpoint_path: "{0}"
            all_model_checkpoint_paths: "{0}"
            """.format(checkpoint_prefix)
        )

    # Add the `classes.json` file.
    if classes_path:
        shutil.copy2(classes_path, path)

    # Store the new checkpoint into the checkpoint index.
    # TODO: Collect metadata correctly.
    metadata = {
        'id': checkpoint_id,
        'status': 'LOCAL',
        'description': 'Description',
        'dataset': {'name': 'COCO'},
        'model': {'name': config.model.type},
    }

    if alias:
        metadata['alias'] = alias

    db = read_checkpoint_db()
    db['checkpoints'].append(metadata)
    save_checkpoint_db(db)

    click.echo('Checkpoint {} created successfully.'.format(checkpoint_id))


@click.command(help='Remove a checkpoint from the index and delete its files.')
@click.argument('id_or_alias')
def delete(id_or_alias):
    db = read_checkpoint_db()
    checkpoint = get_checkpoint(db, id_or_alias)
    if not checkpoint:
        click.echo(
            "Checkpoint '{}' not found in index.".format(id_or_alias)
        )
        return

    # Remove entry from index.
    db['checkpoints'] = [
        cp for cp in db['checkpoints']
        if not cp['id'] == checkpoint['id']
    ]
    save_checkpoint_db(db)

    # Delete tar file associated to checkpoint.
    # TODO: Don't calculate this everytime, centralized way to access.
    path = os.path.join(LUMINOTH_PATH, CHECKPOINT_PATH, checkpoint['id'])
    try:
        shutil.rmtree(path)
    except OSError:
        # The tar is not present, warn the user just in case.
        click.echo(
            'Skipping files deletion; not present in {}.'.format(path)
        )

    click.echo('Checkpoint {} deleted successfully.'.format(checkpoint['id']))


@click.command(help='Export a checkpoint to a tar file for easy sharing.')
@click.argument('id_or_alias')
@click.option('--output', default='.', help="Specify the output location.")
def export(id_or_alias, output):
    db = read_checkpoint_db()
    checkpoint = get_checkpoint(db, id_or_alias)
    if not checkpoint:
        click.echo(
            "Checkpoint '{}' not found in index.".format(id_or_alias)
        )
        return

    # Create the tar that will contain the checkpoint.
    tar_path = os.path.join(
        os.path.abspath(output),
        '{}.tar'.format(checkpoint['id'])
    )
    # TODO: Same as above, no hard-coding.
    checkpoint_path = os.path.join(
        LUMINOTH_PATH, CHECKPOINT_PATH, checkpoint['id']
    )
    with tarfile.open(tar_path, 'w') as f:
        # Add the config file. Dump the dict into a BytesIO, go to the
        # beginning of the file and pass it as a file to the tar.
        # TODO: Python 2 compatibility.
        metadata_file = six.BytesIO()
        metadata_file.write(json.dumps(checkpoint).encode('utf-8'))
        metadata_file.seek(0)

        tarinfo = tarfile.TarInfo(name='metadata.json')
        tarinfo.size = len(metadata_file.getvalue())
        f.addfile(tarinfo=tarinfo, fileobj=metadata_file)

        # Add the files present in the checkpoint's directory.
        for filename in os.listdir(checkpoint_path):
            path = os.path.join(checkpoint_path, filename)
            f.add(path, filename)

    click.echo('Checkpoint {} exported successfully.'.format(checkpoint['id']))


@click.command(help='Import a checkpoint tar into the local index.')
@click.argument('path')
def import_(path):
    # Load the checkpoint metadata first.
    try:
        with tarfile.open(path) as f:
            metadata = json.load(f.extractfile('metadata.json'))
    except tarfile.ReadError:
        click.echo("Invalid file. Is it an exported checkpoint?")
        return
    except KeyError:
        click.echo(
            "Tar file doesn't contain `metadata.json`. "
            "Is it an exported checkpoint?"
        )
        return

    # Check if checkpoint isn't present already.
    # TODO: Check for alias conflict too. Flag to overwrite?
    db = read_checkpoint_db()
    checkpoint = get_checkpoint(db, metadata['id'])
    if checkpoint:
        click.echo(
            "Checkpoint '{}' already found in index.".format(metadata['id'])
        )
        return

    # Check if the output directory doesn't exist already.
    # TODO: Path management.
    output_path = os.path.join(LUMINOTH_PATH, CHECKPOINT_PATH, metadata['id'])
    if os.path.exists(output_path):
        click.echo(
            "Checkpoint directory '{}' for checkpoint_id '{}' already exists. "
            "Try issuing a `lumi checkpoint delete` or delete the directory "
            "manually.".format(output_path, metadata['id'])
        )
        return

    # Extract all the files except `metadata.json` into the checkpoint
    # directory.
    with tarfile.open(path) as f:
        members = [m for m in f.getmembers() if m.name != 'metadata.json']
        f.extractall(output_path, members)

    # Store metadata into the checkpoint index.
    db['checkpoints'].append(metadata)
    save_checkpoint_db(db)

    click.echo('Checkpoint {} imported successfully.'.format(metadata['id']))


@click.command(help='Download checkpoint')
@click.argument('id_or_alias')
def download(id_or_alias):
    click.echo('Not implemented yet.')


@click.group(help='Groups of commands to manage checkpoints')
def checkpoint():
    pass


checkpoint.add_command(create)
checkpoint.add_command(delete)
checkpoint.add_command(download)
checkpoint.add_command(export)
checkpoint.add_command(import_, name='import')
checkpoint.add_command(info)
checkpoint.add_command(list)
