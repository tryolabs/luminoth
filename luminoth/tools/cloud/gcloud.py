import click
import googleapiclient.discovery as discovery
import json
import os
import shutil
import subprocess
import sys
import tempfile
import tensorflow as tf
import time

from datetime import datetime

from google.cloud import storage
from googleapiclient.errors import HttpError
from oauth2client import service_account

from luminoth.utils.config import get_config, dump_config
from luminoth.utils.experiments import save_run


RUNTIME_VERSION = '1.4'
SCALE_TIERS = ['BASIC', 'STANDARD_1', 'PREMIUM_1', 'BASIC_GPU', 'CUSTOM']
MACHINE_TYPES = [
    'standard', 'large_model', 'complex_model_s', 'complex_model_m',
    'complex_model_l', 'standard_gpu', 'complex_model_m_gpu',
    'complex_model_l_gpu', 'standard_p100', 'complex_model_m_p100',
]

DEFAULT_SCALE_TIER = 'BASIC_GPU'
DEFAULT_MASTER_TYPE = 'standard_gpu'
DEFAULT_WORKER_TYPE = 'standard_gpu'
DEFAULT_WORKER_COUNT = 2
DEFAULT_PS_TYPE = 'large_model'
DEFAULT_PS_COUNT = 0

DEFAULT_CONFIG_FILENAME = 'config.yml'
DEFAULT_PACKAGES_PATH = 'packages'


@click.group(help='Train models in Google Cloud ML')
def gc():
    pass


def build_package(bucket, base_path):
    package_path = os.path.abspath(
        os.path.join(os.path.realpath(__file__), '..', '..', '..', '..')
    )

    click.echo('Building custom Luminoth package from "{}".'.format(
        package_path
    ))

    temp_dir = tempfile.mkdtemp()
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

    path = upload_file(
        bucket, '{}/{}'.format(base_path, DEFAULT_PACKAGES_PATH), tarball_path
    )

    shutil.rmtree(temp_dir)

    return path


def get_account_attribute(service_account_json, attr):
    return json.load(
        tf.gfile.GFile(service_account_json, 'r')
    ).get(attr)


def get_project_id(service_account_json):
    return get_account_attribute(service_account_json, 'project_id')


def get_client_id(service_account_json):
    return get_account_attribute(service_account_json, 'client_id')


def get_bucket(service_account_json, bucket_name):
    storage_client = storage.Client.from_service_account_json(
        service_account_json)
    bucket = storage_client.lookup_bucket(bucket_name)
    if not bucket:
        bucket = storage_client.create_bucket(bucket_name)
    return bucket


def upload_data(bucket, file_path, data):
    blob = bucket.blob(file_path)
    blob.upload_from_string(data)
    return blob


def upload_file(bucket, base_path, file_path):
    filename = os.path.basename(file_path)
    path = '{}/{}'.format(base_path, filename)
    click.echo('Uploading file: "{}"\n          -> to "gs://{}/{}"'.format(
        filename, bucket.name, path))
    blob = bucket.blob(path)
    blob.upload_from_file(tf.gfile.GFile(file_path, 'rb'))
    return path


def get_credentials(file):
    return service_account.ServiceAccountCredentials.from_json_keyfile_name(
        file
    )


def cloud_service(credentials, service, version='v1'):
    return discovery.build(service, version, credentials=credentials)


def validate_region(region, project_id, credentials):
    cloudcompute = cloud_service(credentials, 'compute')

    regionrequest = cloudcompute.regions().get(
        region=region, project=project_id
    )
    try:
        regionrequest.execute()
    except HttpError as err:
        if err.resp.status == 404:
            click.echo(
                'Error: Couldn\'t find region "{}" for project "{}".'.format(
                    region, project_id))
        elif err.resp.status == 403:
            click.echo('Error: Forbidden access to resources.')
            click.echo('Raw response:\n{}'.format(err.content))
            click.echo(
                'Make sure to enable "Cloud Compute API", "ML Engine" and '
                '"Storage" for project.')
        else:
            click.echo('Unknown error: {}'.format(err.resp))
        sys.exit(1)


@gc.command(help='Start a training job')
@click.option('--job-id', help='JobId for saving models and logs.')
@click.option('--service-account-json', required=True)
@click.option('--bucket', 'bucket_name', help='Where to save models and logs.')  # noqa
@click.option('--region', default='us-central1', help='Region in which to run the job.')  # noqa
@click.option('--dataset', help='Bucket where the dataset is located.')  # noqa
@click.option('config_files', '--config', '-c', required=True, multiple=True, help='Path to config to use in training.')  # noqa
@click.option('--scale-tier', default=DEFAULT_SCALE_TIER, type=click.Choice(SCALE_TIERS))  # noqa
@click.option('--master-type', default=DEFAULT_MASTER_TYPE, type=click.Choice(MACHINE_TYPES))  # noqa
@click.option('--worker-type', default=DEFAULT_WORKER_TYPE, type=click.Choice(MACHINE_TYPES))  # noqa
@click.option('--worker-count', default=DEFAULT_WORKER_COUNT, type=int)
@click.option('--parameter-server-type', default=DEFAULT_PS_TYPE, type=click.Choice(MACHINE_TYPES))  # noqa
@click.option('--parameter-server-count', default=DEFAULT_PS_COUNT, type=int)
def train(job_id, service_account_json, bucket_name, region, config_files,
          dataset, scale_tier, master_type, worker_type, worker_count,
          parameter_server_type, parameter_server_count):

    project_id = get_project_id(service_account_json)
    if project_id is None:
        raise ValueError(
            'Missing "project_id" in service_account_json "{}"'.format(
                service_account_json))

    if bucket_name is None:
        client_id = get_client_id(service_account_json)
        bucket_name = 'luminoth-{}'.format(client_id)
        click.echo(
            'Bucket name not specified. Using "{}".'.format(bucket_name))

    credentials = get_credentials(service_account_json)
    validate_region(region, project_id, credentials)

    # Creates bucket for logs and models if it doesn't exist
    bucket = get_bucket(service_account_json, bucket_name)

    if not job_id:
        job_id = 'train_{}'.format(datetime.now().strftime("%Y%m%d_%H%M%S"))

    # Define path in bucket to store job's config, logs, etc.
    base_path = 'lumi_{}'.format(job_id)

    package_path = build_package(bucket, base_path)
    job_dir = 'gs://{}/{}/'.format(bucket_name, base_path)

    override_params = [
        'train.job_dir={}'.format(job_dir),
    ]

    if dataset:
        # Check if absolute or relative dataset path
        if not dataset.startswith('gs://'):
            dataset = 'gs://{}'.format(dataset)
        override_params.append('dataset.dir={}'.format(dataset))

    config = get_config(config_files, override_params=override_params)
    # We should validate config before submitting job

    # Update final config file to job bucket
    config_path = os.path.join(base_path, DEFAULT_CONFIG_FILENAME)
    upload_data(bucket, config_path, dump_config(config))

    args = ['--config', os.path.join(job_dir, DEFAULT_CONFIG_FILENAME)]

    cloudml = cloud_service(credentials, 'ml')

    training_inputs = {
        'scaleTier': scale_tier,
        'packageUris': [
            'gs://{}/{}'.format(bucket_name, package_path)
        ],
        'pythonModule': 'luminoth.train',
        'args': args,
        'region': region,
        'jobDir': job_dir,
        'runtimeVersion': RUNTIME_VERSION,
    }

    if scale_tier == 'CUSTOM':
        training_inputs['masterType'] = master_type
        if worker_count > 0:
            training_inputs['workerCount'] = worker_count
            training_inputs['workerType'] = worker_type

        if parameter_server_count > 0:
            training_inputs['parameterServerCount'] = parameter_server_count
            training_inputs['parameterServerType'] = parameter_server_type

    job_spec = {
        'jobId': job_id,
        'trainingInput': training_inputs
    }

    jobrequest = cloudml.projects().jobs().create(
        body=job_spec, parent='projects/{}'.format(project_id))

    try:
        click.echo('Submitting training job.')
        res = jobrequest.execute()
        click.echo('Job {} submitted successfully.'.format(job_id))
        click.echo('state = {}, createTime = {}'.format(
            res.get('state'), res.get('createTime')))

        save_run(config, environment='gcloud', extra_config=job_spec)

    except Exception as err:
        click.echo(
            'There was an error creating the training job. '
            'Check the details: \n{}'.format(err._get_reason())
        )


@gc.command(help='Start a evaluation job')
@click.option('--job-id', required=True, help='JobId for saving models and logs.')  # noqa
@click.option('--service-account-json', required=True)
@click.option('--bucket', 'bucket_name', help='Where to save models and logs.')  # noqa
@click.option('dataset_split', '--split', default='val', help='Dataset split to use.')  # noqa
@click.option('--region', default='us-central1', help='Region in which to run the job.')  # noqa
@click.option('--machine-type', default=DEFAULT_MASTER_TYPE, type=click.Choice(MACHINE_TYPES))  # noqa
@click.option('--rebuild', default=False, is_flag=True, help='Rebuild and upload package.')  # noqa
@click.option('--postfix', default='eval', help='Postfix for the evaluation job name.')  # noqa
def evaluate(job_id, service_account_json, bucket_name, dataset_split, region,
             machine_type, rebuild, postfix):
    project_id = get_project_id(service_account_json)
    if project_id is None:
        raise ValueError(
            'Missing "project_id" in service_account_json "{}"'.format(
                service_account_json))

    if bucket_name is None:
        client_id = get_client_id(service_account_json)
        bucket_name = 'luminoth-{}'.format(client_id)
        click.echo(
            'Bucket name not specified. Using "{}".'.format(bucket_name))

    credentials = get_credentials(service_account_json)
    validate_region(region, project_id, credentials)
    job_folder = 'lumi_{}'.format(job_id)

    if rebuild:
        bucket = get_bucket(service_account_json, bucket_name)
        build_package(bucket, job_folder)

    job_dir = 'gs://{}/{}'.format(bucket_name, job_folder)

    config_path = '{}/{}'.format(job_dir, DEFAULT_CONFIG_FILENAME)
    package_dir = '{}/{}'.format(job_dir, DEFAULT_PACKAGES_PATH)

    package_files = tf.gfile.ListDirectory(package_dir)
    package_filename = [n for n in package_files if n.endswith('tar.gz')][0]
    package_path = '{}/{}'.format(package_dir, package_filename)

    cloudml = cloud_service(credentials, 'ml')

    args = [
        ['--config', config_path],
        ['--split', dataset_split],
    ]

    training_inputs = {
        'scaleTier': 'CUSTOM',
        'masterType': machine_type,
        'packageUris': [package_path],
        'pythonModule': 'luminoth.eval',
        'args': args,
        'region': region,
        'jobDir': job_dir,
        'runtimeVersion': RUNTIME_VERSION,
    }

    evaluate_job_id = '{}_{}'.format(job_id, postfix)
    job_spec = {
        'jobId': evaluate_job_id,
        'trainingInput': training_inputs
    }

    jobrequest = cloudml.projects().jobs().create(
        body=job_spec, parent='projects/{}'.format(project_id))

    try:
        click.echo('Submitting evaluation job.')
        res = jobrequest.execute()
        click.echo('Job {} submitted successfully.'.format(evaluate_job_id))
        click.echo('state = {}, createTime = {}'.format(
            res.get('state'), res.get('createTime')))

    except Exception as err:
        click.echo(
            'There was an error creating the evaluation job. '
            'Check the details: \n{}'.format(err._get_reason())
        )


@gc.command(help='List project jobs')
@click.option('--service-account-json', required=True)
@click.option('--running', is_flag=True, help='List only jobs that are running.')  # noqa
def jobs(service_account_json, running):
    project_id = get_project_id(service_account_json)
    if project_id is None:
        raise ValueError(
            'Missing "project_id" in service_account_json "{}"'.format(
                service_account_json))

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
@click.option('--service-account-json', required=True)
@click.option('--polling-interval', default=60, help='Polling interval in seconds.')  # noqa
def logs(job_id, service_account_json, polling_interval):
    project_id = get_project_id(service_account_json)
    if project_id is None:
        raise ValueError(
            'Missing "project_id" in service_account_json "{}"'.format(
                service_account_json))

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
