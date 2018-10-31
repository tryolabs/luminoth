import click
import json
import os
import shutil
import six
import requests
import tarfile
import tempfile
import tensorflow as tf
import uuid

from datetime import datetime

from luminoth import __version__ as lumi_version
from luminoth.utils.config import get_config
from luminoth.utils.homedir import get_luminoth_home


CHECKPOINT_INDEX = 'checkpoints.json'
CHECKPOINT_PATH = 'checkpoints'
REMOTE_INDEX_URL = (
    'https://github.com/tryolabs/luminoth/releases/download/v0.0.3/'
    'checkpoints.json'
)


# Definition of path management functions.

def get_checkpoints_directory():
    """Returns checkpoint directory within Luminoth's homedir."""
    # Checkpoint directory, `$LUMI_HOME/checkpoints/`. Create if not present.
    path = os.path.join(get_luminoth_home(), CHECKPOINT_PATH)
    if not os.path.exists(path):
        tf.gfile.MakeDirs(path)
    return path


def get_checkpoint_path(checkpoint_id):
    """Returns checkpoint's directory."""
    return os.path.join(get_checkpoints_directory(), checkpoint_id)


# Index-related functions: access and mutation.

def read_checkpoint_db():
    """Reads the checkpoints database file from disk."""
    path = os.path.join(get_checkpoints_directory(), CHECKPOINT_INDEX)
    if not os.path.exists(path):
        return {'checkpoints': []}

    with open(path) as f:
        index = json.load(f)

    return index


def save_checkpoint_db(checkpoints):
    """Overwrites the database file in disk with `checkpoints`."""
    path = os.path.join(get_checkpoints_directory(), CHECKPOINT_INDEX)
    with open(path, 'w') as f:
        json.dump(checkpoints, f)


def merge_index(local_index, remote_index):
    """Merge the `remote_index` into `local_index`.

    The merging process is only applied over the checkpoints in `local_index`
    marked as ``remote``.
    """

    non_remotes_in_local = [
        c for c in local_index['checkpoints']
        if c['source'] != 'remote'
    ]
    remotes_in_local = {
        c['id']: c for c in local_index['checkpoints']
        if c['source'] == 'remote'
    }

    to_add = []
    seen_ids = set()
    for checkpoint in remote_index['checkpoints']:
        seen_ids.add(checkpoint['id'])
        local = remotes_in_local.get(checkpoint['id'])
        if local:
            # Checkpoint is in local index. Overwrite all the fields.
            local.update(**checkpoint)
        elif not local:
            # Checkpoint not found, it's an addition. Transform into our schema
            # before appending to `to_add`. (The remote index schema is exactly
            # the same except for the ``source`` and ``status`` keys.)
            checkpoint['source'] = 'remote'
            checkpoint['status'] = 'NOT_DOWNLOADED'
            to_add.append(checkpoint)

    # Out of the removed checkpoints, only keep those with status
    # ``DOWNLOADED`` and turn them into local checkpoints.
    missing_ids = set(remotes_in_local.keys()) - seen_ids
    already_downloaded = [
        c for c in remotes_in_local.values()
        if c['id'] in missing_ids and c['status'] == 'DOWNLOADED'
    ]
    for checkpoint in already_downloaded:
        checkpoint['status'] = 'LOCAL'
        checkpoint['source'] = 'local'

    new_remotes = [
        c for c in remotes_in_local.values()
        if not c['id'] in missing_ids  # Checkpoints to remove.
    ] + to_add + already_downloaded

    if len(to_add):
        click.echo('{} new remote checkpoints added.'.format(len(to_add)))
    if len(missing_ids):
        if len(already_downloaded):
            click.echo('{} remote checkpoints transformed to local.'.format(
                len(already_downloaded)
            ))
        click.echo('{} remote checkpoints removed.'.format(
            len(missing_ids) - len(already_downloaded)
        ))
    if not len(to_add) and not len(missing_ids):
        click.echo('No changes in remote index.')

    local_index['checkpoints'] = non_remotes_in_local + new_remotes

    return local_index


def get_checkpoint(db, id_or_alias):
    """Returns checkpoint entry in `db` indicated by `id_or_alias`.

    First tries to match an ID, then an alias. For the case of repeated
    aliases, will first match local checkpoints and then remotes. In both
    cases, matching will be newest first.
    """
    # Go through the checkpoints ordered by creation date. There shouldn't be
    # repeated aliases, but if there are, prioritize the newest one.
    local_checkpoints = sorted(
        [c for c in db['checkpoints'] if c['source'] == 'local'],
        key=lambda c: c['created_at'], reverse=True
    )
    remote_checkpoints = sorted(
        [c for c in db['checkpoints'] if c['source'] == 'remote'],
        key=lambda c: c['created_at'], reverse=True
    )

    selected = []
    for cp in local_checkpoints:
        if cp['id'] == id_or_alias or cp['alias'] == id_or_alias:
            selected.append(cp)

    for cp in remote_checkpoints:
        if cp['id'] == id_or_alias or cp['alias'] == id_or_alias:
            selected.append(cp)

    if len(selected) < 1:
        return None

    if len(selected) > 1:
        click.echo(
            "Multiple checkpoints found for '{}' ({}). Returning '{}'.".format(
                id_or_alias, len(selected), selected[0]['id']
            )
        )

    return selected[0]


def get_checkpoint_config(id_or_alias, prompt=True):
    """Returns the checkpoint config object in order to load the model.

    If `prompt` is ``True`` and the checkpoint is not present in the index,
    prompt the user to refresh the index. If the checkpoint is present in the
    index but is remote and not yet downloaded, prompt to download.
    """
    db = read_checkpoint_db()
    checkpoint = get_checkpoint(db, id_or_alias)

    if prompt and not checkpoint:
        # Checkpoint not found in database. Prompt for refreshing the index and
        # try again.
        click.confirm(
            'Checkpoint not found. Check remote repository?', abort=True
        )
        db = refresh_remote_index()
        checkpoint = get_checkpoint(db, id_or_alias)
        if not checkpoint:
            # Still not found, abort.
            click.echo(
                "Checkpoint isn't available in remote repository either."
            )
            raise ValueError('Checkpoint not found.')
    elif not checkpoint:
        # No checkpoint but didn't prompt.
        raise ValueError('Checkpoint not found.')

    if prompt and checkpoint['status'] == 'NOT_DOWNLOADED':
        # Checkpoint hasn't been downloaded yet. Prompt for downloading it
        # before continuing.
        click.confirm(
            'Checkpoint not present locally. Want to download it?', abort=True
        )
        download_remote_checkpoint(db, checkpoint)
    elif checkpoint['status'] == 'NOT_DOWNLOADED':
        # Not downloaded but didn't prompt.
        raise ValueError('Checkpoint not downloaded.')

    path = get_checkpoint_path(checkpoint['id'])
    config = get_config(os.path.join(path, 'config.yml'))

    # Config paths should point to the path where the checkpoint files are
    # stored.
    config.dataset.dir = path
    config.train.job_dir = get_checkpoints_directory()

    return config


def field_allowed(field):
    return field in [
        'name', 'description', 'alias', 'dataset.name', 'dataset.num_classes',
    ]


def parse_entries(entries, editing=False):
    """Parses and validates `entries`."""
    # Parse the entries (list of `<field>=<value>`).
    values = [tuple(entry.split('=')) for entry in entries]

    # Validate allowed field values to modify.
    disallowed = [k for k, _ in values if not field_allowed(k)]
    if disallowed:
        click.echo("The following fields may not be set: {}".format(
            ', '.join(disallowed)
        ))
        return

    # Make sure there are no repeated values.
    unique = set([k for k, _ in values])
    if len(values) - len(unique) > 0:
        click.echo("Repeated fields. Each field may be passed exactly once.")
        return

    return dict(values)


def apply_entries(checkpoint, entries):
    """Recursively modifies `checkpoint` with `entries` values."""
    for field, value in entries.items():
        # Access the last dict (if nested) and modify it.
        to_edit = checkpoint
        bits = field.split('.')
        for bit in bits[:-1]:
            to_edit = to_edit[bit]
        to_edit[bits[-1]] = value

    return checkpoint


# Network-related IO functions.

def get_remote_index_url():
    url = REMOTE_INDEX_URL
    if 'LUMI_REMOTE_URL' in os.environ:
        url = os.environ['LUMI_REMOTE_URL']
    return url


def fetch_remote_index():
    url = get_remote_index_url()
    index = requests.get(url).json()
    return index


def refresh_remote_index():
    click.echo('Retrieving remote index... ', nl=False)
    remote = fetch_remote_index()
    click.echo('done.')

    db = read_checkpoint_db()
    db = merge_index(db, remote)

    save_checkpoint_db(db)

    return db


def download_remote_checkpoint(db, checkpoint):
    # Check if output directory doesn't exist already to fail early.
    output = get_checkpoint_path(checkpoint['id'])
    if os.path.exists(output):
        click.echo(
            "Checkpoint directory '{}' for checkpoint_id '{}' already exists. "
            "Try issuing a `lumi checkpoint delete` or delete the directory "
            "manually.".format(output, checkpoint['id'])
        )
        return

    # Create a temporary directory to download the tar into.
    tempdir = tempfile.mkdtemp()
    path = os.path.join(tempdir, '{}.tar'.format(checkpoint['id']))

    # Start the actual tar file download.
    response = requests.get(checkpoint['url'], stream=True)
    length = int(response.headers.get('Content-Length'))
    chunk_size = 16 * 1024
    progressbar = click.progressbar(
        response.iter_content(chunk_size=chunk_size),
        length=length / chunk_size, label='Downloading checkpoint...',
    )

    with open(path, 'wb') as f:
        with progressbar as content:
            for chunk in content:
                f.write(chunk)

    # Import the checkpoint from the tar.
    click.echo("Importing checkpoint... ", nl=False)
    with tarfile.open(path) as f:
        members = [m for m in f.getmembers() if m.name != 'metadata.json']
        f.extractall(output, members)
    click.echo("done.")

    # Update the checkpoint status and persist database.
    checkpoint['status'] = 'DOWNLOADED'
    save_checkpoint_db(db)

    # And finally make sure to delete the temp dir.
    shutil.rmtree(tempdir)

    click.echo("Checkpoint imported successfully.")


# Actual command definition.

@click.command(help='List available checkpoints.')
def list():
    db = read_checkpoint_db()

    if not db['checkpoints']:
        click.echo('No checkpoints available.')
        return

    template = '| {:>12} | {:>21} | {:>11} | {:>6} | {:>14} |'

    header = template.format(
        'id', 'name', 'alias', 'source', 'status'
    )
    click.echo('=' * len(header))
    click.echo(header)
    click.echo('=' * len(header))

    for checkpoint in db['checkpoints']:
        line = template.format(
            checkpoint['id'],
            checkpoint['name'],
            checkpoint['alias'],
            checkpoint['source'],
            checkpoint['status'],
        )
        click.echo(line)

    click.echo('=' * len(header))


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

    if checkpoint['alias']:
        click.echo('{} ({}, {})'.format(
            checkpoint['name'], checkpoint['id'], checkpoint['alias']
        ))
    else:
        click.echo('{} ({})'.format(checkpoint['name'], checkpoint['id']))

    if checkpoint['description']:
        click.echo(checkpoint['description'])

    click.echo()

    click.echo('Model used: {}'.format(checkpoint['model']))
    click.echo('Dataset information')
    click.echo('    Name: {}'.format(checkpoint['dataset']['name']))
    click.echo('    Number of classes: {}'.format(
        checkpoint['dataset']['num_classes']
    ))

    click.echo()

    click.echo('Creation date: {}'.format(checkpoint['created_at']))
    click.echo('Luminoth version: {}'.format(checkpoint['luminoth_version']))

    click.echo()

    if checkpoint['source'] == 'remote':
        click.echo('Source: {} ({})'.format(
            checkpoint['source'], checkpoint['status']
        ))
        click.echo('URL: {}'.format(checkpoint['url']))
    else:
        click.echo('Source: {}'.format(checkpoint['source']))


@click.command(help='Create a checkpoint from a configuration file.')
@click.argument('config_files', nargs=-1)
@click.option(
    'override_params', '--override', '-o', multiple=True,
    help='Override model config params.'
)
@click.option(
    'entries', '--entry', '-e', multiple=True,
    help="Specify checkpoint's metadata field value."
)
def create(config_files, override_params, entries):
    # Parse the entries passed as options.
    entries = parse_entries(entries)
    if entries is None:
        return

    click.echo('Creating checkpoint for given configuration...')
    # Get and build the configuration file for the model.
    config = get_config(config_files, override_params=override_params)

    # Retrieve the files for the last checkpoint available.
    run_dir = os.path.join(config.train.job_dir, config.train.run_name)
    ckpt = tf.train.get_checkpoint_state(run_dir)
    if not ckpt or not ckpt.all_model_checkpoint_paths:
        click.echo("Couldn't find checkpoint in '{}'.".format(run_dir))
        return

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
    config.dataset.dir = '.'
    config.train.job_dir = '.'
    config.train.run_name = checkpoint_id

    # Create the directory that will contain the model.
    path = get_checkpoint_path(checkpoint_id)
    tf.gfile.MakeDirs(path)

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

    # Add the `classes.json` file. Also get the number of classes, if
    # available.
    num_classes = None
    if classes_path:
        shutil.copy2(classes_path, path)
        with open(classes_path) as f:
            num_classes = len(json.load(f))

    # Store the new checkpoint into the checkpoint index.
    metadata = {
        'id': checkpoint_id,
        'name': entries.get('name', ''),
        'description': entries.get('description', ''),
        'alias': entries.get('alias', ''),

        'model': config.model.type,
        'dataset': {
            'name': entries.get('dataset.name', ''),
            'num_classes': (
                num_classes or entries.get('dataset.num_classes', None)
            ),
        },

        'luminoth_version': lumi_version,
        'created_at': datetime.utcnow().isoformat(),

        'status': 'LOCAL',
        'source': 'local',
        'url': None,  # Only for remotes.
    }

    db = read_checkpoint_db()
    db['checkpoints'].append(metadata)
    save_checkpoint_db(db)

    click.echo('Checkpoint {} created successfully.'.format(checkpoint_id))


@click.command(help="Edit a checkpoint's metadata.")
@click.argument('id_or_alias')
@click.option(
    'entries', '--entry', '-e', multiple=True,
    help="Overwrite checkpoint's metadata field value."
)
def edit(id_or_alias, entries):
    db = read_checkpoint_db()
    checkpoint = get_checkpoint(db, id_or_alias)
    if not checkpoint:
        click.echo(
            "Checkpoint '{}' not found in index.".format(id_or_alias)
        )
        return

    # Parse the entries to modify and apply them.
    entries = parse_entries(entries, editing=True)
    if entries is None:
        return

    apply_entries(checkpoint, entries)

    save_checkpoint_db(db)

    click.echo('Checkpoint edited successfully.')


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

    # If checkpoint is local, remove entry from index. If it's remote, only
    # mark as ``NOT_DOWNLOADED``.
    if checkpoint['source'] == 'local':
        db['checkpoints'] = [
            cp for cp in db['checkpoints']
            if not cp['id'] == checkpoint['id']
        ]
    else:  # Remote.
        if checkpoint['status'] == 'NOT_DOWNLOADED':
            click.echo("Checkpoint isn't downloaded. Nothing to delete.")
            return
        checkpoint['status'] = 'NOT_DOWNLOADED'
    save_checkpoint_db(db)

    # Delete tar file associated to checkpoint.
    path = get_checkpoint_path(checkpoint['id'])
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
@click.option('--output', default='.', help='Specify the output location.')
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
        os.path.abspath(os.path.expanduser(output)),
        '{}.tar'.format(checkpoint['id'])
    )
    checkpoint_path = get_checkpoint_path(checkpoint['id'])
    with tarfile.open(tar_path, 'w') as f:
        # Add the config file. Dump the dict into a BytesIO, go to the
        # beginning of the file and pass it as a file to the tar.
        metadata_file = six.BytesIO()
        metadata_file.write(json.dumps(checkpoint).encode('utf-8'))
        metadata_file.seek(0)

        tarinfo = tarfile.TarInfo(name='metadata.json')
        tarinfo.size = len(metadata_file.getvalue())
        f.addfile(tarinfo=tarinfo, fileobj=metadata_file)

        metadata_file.close()

        # Add the files present in the checkpoint's directory.
        for filename in os.listdir(checkpoint_path):
            path = os.path.join(checkpoint_path, filename)
            f.add(path, filename)

    click.echo('Checkpoint {} exported successfully.'.format(checkpoint['id']))


@click.command(help='Import a checkpoint tar into the local index.')
@click.argument('path')
def import_(path):
    # Load the checkpoint metadata first.
    path = os.path.expanduser(path)
    try:
        with tarfile.open(path) as f:
            # This way of loading the JSON works in all supported versions of
            # Python 2 and 3.
            # See: https://github.com/tryolabs/luminoth/issues/222
            metadata = json.loads(
                f.extractfile('metadata.json').read().decode('utf-8')
            )
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
    db = read_checkpoint_db()
    checkpoint = get_checkpoint(db, metadata['id'])
    if checkpoint:
        click.echo(
            "Checkpoint '{}' already found in index.".format(metadata['id'])
        )
        return

    # Check if the output directory doesn't exist already.
    output_path = get_checkpoint_path(metadata['id'])
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


@click.command(help='Refresh the remote checkpoint index.')
def refresh():
    refresh_remote_index()


@click.command(help='Download a remote checkpoint.')
@click.argument('id_or_alias')
def download(id_or_alias):
    db = read_checkpoint_db()
    checkpoint = get_checkpoint(db, id_or_alias)
    if not checkpoint:
        click.echo(
            "Checkpoint '{}' not found in index.".format(id_or_alias)
        )
        return

    if checkpoint['source'] != 'remote':
        click.echo(
            "Checkpoint is not remote. If you intended to download a remote "
            "checkpoint and used an alias, try using the id directly."
        )
        return

    if checkpoint['status'] != 'NOT_DOWNLOADED':
        click.echo("Checkpoint is already downloaded.")
        return

    download_remote_checkpoint(db, checkpoint)


@click.group(help='Groups of commands to manage checkpoints')
def checkpoint():
    pass


checkpoint.add_command(create)
checkpoint.add_command(delete)
checkpoint.add_command(download)
checkpoint.add_command(edit)
checkpoint.add_command(export)
checkpoint.add_command(import_, name='import')
checkpoint.add_command(info)
checkpoint.add_command(list)
checkpoint.add_command(refresh)
