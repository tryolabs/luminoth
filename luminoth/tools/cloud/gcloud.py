import click
import os
import tempfile
import time
import tensorflow as tf
import googleapiclient.discovery as discovery
import subprocess

from datetime import datetime

from google.cloud import storage
from oauth2client import service_account


SCALE_TIERS = ['BASIC', 'STANDARD_1', 'PREMIUM_1', 'BASIC_GPU', 'CUSTOM']
MACHINE_TYPES = [
    'standard', 'large_model', 'complex_model_s', 'complex_model_m',
    'complex_model_l', 'standard_gpu', 'complex_model_m_gpu',
    'complex_model_l_gpu'
]

DEFAULT_SCALE_TIER = 'BASIC_GPU'
DEFAULT_MASTER_TYPE = 'standard_gpu'
DEFAULT_WORKER_TYPE = 'standard_gpu'
DEFAULT_WORKER_COUNT = 2


@click.group(help='Train models in Google Cloud ML')
def gc():
    pass


def build_package(bucket):
    package_path = os.path.abspath(
        os.path.join(os.path.realpath(__file__), '..', '..', '..', '..')
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = os.path.join(temp_dir, 'output')

        devnull = open(os.devnull, 'w')
        subprocess.call(
            [
                'python', 'setup.py', 'egg_info', '--egg-base', temp_dir,
                'build', '--build-base', temp_dir, '--build-temp', temp_dir,
                'sdist', '--dist-dir', output_dir
            ],
            cwd=package_path, stdout=devnull, stderr=devnull
        )
        subprocess.call(
            [
                'python', 'setup.py', 'build', '--build-base', temp_dir,
                '--build-temp', temp_dir, 'sdist', '--dist-dir', output_dir
            ],
            cwd=package_path, stdout=devnull, stderr=devnull
        )
        subprocess.call(
            ['python', 'setup.py', 'sdist', '--dist-dir', output_dir],
            cwd=package_path, stdout=devnull, stderr=devnull
        )

        tarball_filename = os.listdir(output_dir)[0]
        tarball_path = os.path.join(
            output_dir, tarball_filename
        )

        path = upload_file(bucket, 'packages', tarball_path)

        return path


def get_bucket(service_account_json, bucket_name):
    storage_client = storage.Client.from_service_account_json(
        service_account_json)
    bucket = storage_client.lookup_bucket(bucket_name)
    if not bucket:
        bucket = storage_client.create_bucket(bucket_name)
    return bucket


def upload_file(bucket, base_path, filename):
    click.echo('Uploading config file: {}'.format(filename))
    path = '{}/{}'.format(base_path, os.path.basename(filename))
    blob = bucket.blob(path)
    blob.upload_from_file(tf.gfile.GFile(filename, 'rb'))
    return path


def get_credentials(file):
    return service_account.ServiceAccountCredentials.from_json_keyfile_name(
        file
    )


def cloud_service(credentials, service, version='v1'):
    return discovery.build(service, version, credentials=credentials)


@gc.command(help='Start a training job')
@click.option('--job-id', help='JobId for saving models and logs.')
@click.option('--project-id', required=True)
@click.option('--service-account-json', required=True)
@click.option('--bucket', 'bucket_name', required=True, help='Where to save models and logs.')  # noqa
@click.option('--dataset', required=True, help='Bucket where the dataset is located.')  # noqa
@click.option('--config', help='Path to config to use in training.')
@click.option('--scale-tier', default=DEFAULT_SCALE_TIER, type=click.Choice(SCALE_TIERS))  # noqa
@click.option('--master-type', default=DEFAULT_MASTER_TYPE, type=click.Choice(MACHINE_TYPES))  # noqa
@click.option('--worker-type', default=DEFAULT_WORKER_TYPE, type=click.Choice(MACHINE_TYPES))  # noqa
@click.option('--worker-count', default=DEFAULT_WORKER_COUNT, type=int)
def train(job_id, project_id, service_account_json, bucket_name, config,
          dataset, scale_tier, master_type, worker_type, worker_count):
    args = []

    # Creates bucket for logs and models if it doesn't exist
    bucket = get_bucket(service_account_json, bucket_name)

    package_path = build_package(bucket)

    if not job_id:
        job_id = 'train_{}'.format(datetime.now().strftime("%Y%m%d_%H%M%S"))

    # Define path in bucket to store job's config, logs, etc.
    base_path = 'lumi_{}'.format(job_id)

    # Check if absolute or relative dataset path
    if not dataset.startswith('gs://'):
        dataset = 'gs://{}'.format(dataset)

    args.extend([
        '--job-dir', 'gs://{}/{}'.format(bucket_name, base_path),
        '--override', 'dataset.dir={}'.format(dataset),
        # Turning off data_augmentation because of TF 1.2 limitations
        '--override', 'dataset.data_augmentation=false'
    ])

    if config:
        # Upload config file to be used by the training job.
        path = upload_file(bucket, base_path, config)
        args.extend(['--config', 'gs://{}/{}'.format(bucket_name, path)])

    credentials = get_credentials(service_account_json)
    cloudml = cloud_service(credentials, 'ml')

    training_inputs = {
        'scaleTier': scale_tier,
        'packageUris': [
            'gs://{}/{}'.format(bucket_name, package_path)
        ],
        'pythonModule': 'luminoth.train',
        'args': args,
        'region': 'us-central1',
        'jobDir': 'gs://{}/{}/train/'.format(bucket_name, base_path),
        'runtimeVersion': '1.2'
    }

    if scale_tier == 'CUSTOM':
        training_inputs['masterType'] = master_type
        training_inputs['workerType'] = worker_type
        training_inputs['workerCount'] = worker_count

    job_spec = {
        'jobId': job_id,
        'trainingInput': training_inputs
    }

    request = cloudml.projects().jobs().create(
        body=job_spec, parent='projects/{}'.format(project_id))

    try:
        click.echo('Submitting training job.')
        request.execute()
        click.echo('Job {} submitted successfully.'.format(job_id))
    except Exception as err:
        click.echo(
            'There was an error creating the training job. '
            'Check the details: \n{}'.format(err._get_reason())
        )


@gc.command(help='List project jobs')
@click.option('--project-id', required=True)
@click.option('--service-account-json', required=True)
@click.option('--running', is_flag=True, help='List only jobs that are running.')  # noqa
def jobs(project_id, service_account_json, running):
    credentials = get_credentials(service_account_json)
    cloudml = cloud_service(credentials, 'ml')
    request = cloudml.projects().jobs().list(
        parent='projects/{}'.format(project_id))

    try:
        response = request.execute()
        jobs = response['jobs']

        if not jobs:
            click.echo('There are no jobs for this project.')
            return

        if running:
            jobs = [j for j in jobs if j['state'] == 'RUNNING']
            if not jobs:
                click.echo('There are no jobs running.')
                return

        for job in jobs:
            click.echo('Id: {} Created: {} State: {}'.format(
                job['jobId'], job['createTime'], job['state']))
    except Exception as err:
        click.echo(
            'There was an error fetching jobs. '
            'Check the details: \n{}'.format(err._get_reason())
        )


@gc.command(help='Show logs from a running job')
@click.argument('job_id')
@click.option('--project-id', required=True)
@click.option('--service-account-json', required=True)
@click.option('--polling-interval', default=60, help='Polling interval in seconds.')  # noqa
def logs(job_id, project_id, service_account_json, polling_interval):
    credentials = get_credentials(service_account_json)
    cloudlog = cloud_service(credentials, 'logging', 'v2')

    job_filter = 'resource.labels.job_id = "{}"'.format(job_id)
    last_timestamp = None
    while True:
        filters = [job_filter]
        if last_timestamp:
            filters.append('timestamp > "{}"'.format(last_timestamp))

        # Fetch all pages.
        entries = []
        next_page = None
        while True:
            request = cloudlog.entries().list(body={
                'resourceNames': 'projects/{}'.format(project_id),
                'filter': ' AND '.join(filters),
                'pageToken': next_page,
            })

            try:
                response = request.execute()
                next_page = response.get('nextPageToken', None)
                entries.extend(response.get('entries', []))
                if not next_page:
                    break
            except Exception as err:
                click.echo(
                    'There was an error fetching the logs. '
                    'Check the details: \n{}'.format(err._get_reason())
                )
                break

        for entry in entries:
            last_timestamp = entry['timestamp']

            if 'jsonPayload' in entry:
                message = entry['jsonPayload']['message']
            elif 'textPayload' in entry:
                message = entry['textPayload']
            else:
                continue

            click.echo('{:30} :: {:7} :: {}'.format(
                entry['timestamp'], entry['severity'], message.strip()
            ))

        time.sleep(polling_interval)
