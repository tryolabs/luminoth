import click
import os.path
import tensorflow as tf
import googleapiclient.discovery as discovery

from datetime import datetime

from google.cloud import storage
from google.oauth2 import service_account


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


@click.command(help='Start a training job in Google Cloud ML')
@click.option('--job-id', help='JobId for saving models and logs.')
@click.option('--project-id', required=True)
@click.option('--service-account-json', required=True)
@click.option('--bucket', 'bucket_name', required=True, help='Where to save models and logs.')
@click.option('--dataset', required=True, help='Bucket where the dataset is located.')
@click.option('--config')
def gc(job_id, project_id, service_account_json, bucket_name, config, dataset):
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

    credentials = service_account.Credentials.from_service_account_file(
        service_account_json)
    cloudml = discovery.build('ml', 'v1', credentials=credentials)

    training_inputs = {
        'scaleTier': 'BASIC_GPU',
        'packageUris': ['gs://luminoth-config/luminoth-0.0.1-py2-none-any.whl'],
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
