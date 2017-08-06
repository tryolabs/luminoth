import click
import os.path
import time
import tensorflow as tf
import googleapiclient.discovery as discovery

from datetime import datetime

from google.cloud import storage
from google.oauth2 import service_account


@click.group(help='Train models in Google Cloud ML')
def gc():
    pass


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
    return service_account.Credentials.from_service_account_file(file)


def cloud_service(credentials, service, version='v1'):
    return discovery.build(service, version, credentials=credentials)


@gc.command(help='Start a training job')
@click.option('--job-id', help='JobId for saving models and logs.')
@click.option('--project-id', required=True)
@click.option('--service-account-json', required=True)
@click.option('--bucket', 'bucket_name', required=True, help='Where to save models and logs.') # noqa
@click.option('--dataset', required=True, help='Bucket where the dataset is located.') # noqa
@click.option('--config', help='Path to config to use in training.')
def train(job_id, project_id, service_account_json, bucket_name, config,
          dataset):
    args = []

    if not job_id:
        job_id = 'train_{}'.format(datetime.now().strftime("%Y%m%d_%H%M%S"))

    # Define path in bucket to store job's config, logs, etc.
    base_path = 'lumi_{}'.format(job_id)

    # Check if absolute or relative dataset path
    if not dataset.startswith('gs://'):
        dataset = 'gs://{}'.format(dataset)

    args.extend([
        '--log-dir', 'gs://{}/{}/logs'.format(bucket_name, base_path),
        '--model-dir', 'gs://{}/{}/model'.format(bucket_name, base_path),
        '--override', 'dataset.dir={}'.format(dataset)
    ])

    # Creates bucket for logs and models if it doesn't exist
    bucket = get_bucket(service_account_json, bucket_name)

    if config:
        # Upload config file to be used by the training job.
        path = upload_file(bucket, base_path, config)
        args.extend(['--config', 'gs://{}/{}'.format(bucket_name, path)])

    credentials = get_credentials(service_account_json)
    cloudml = cloud_service(credentials, 'ml')

    training_inputs = {
        'scaleTier': 'BASIC_GPU',
        'packageUris': [
            'gs://luminoth-config/luminoth-0.0.1-py2-none-any.whl'
        ],
        'pythonModule': 'luminoth.train',
        'args': args,
        'region': 'us-central1',
        'jobDir': 'gs://{}/{}/train/'.format(bucket_name, base_path),
        'runtimeVersion': '1.2'
    }

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
