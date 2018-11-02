import click
import json
import os
import shutil
import subprocess
import sys
import tempfile
import tensorflow as tf
import time

from functools import wraps
from datetime import datetime

from luminoth.utils.config import get_config, dump_config
from luminoth.utils.experiments import save_run

MISSING_DEPENDENCIES = False
try:
    from google.cloud import storage
    from googleapiclient import discovery, errors
    from oauth2client import service_account
except ImportError:
    MISSING_DEPENDENCIES = True


RUNTIME_VERSION = '1.9'
SCALE_TIERS = [
    'BASIC', 'STANDARD_1', 'PREMIUM_1', 'BASIC_GPU', 'BASIC_TPU', 'CUSTOM'
]
MACHINE_TYPES = [
    'standard', 'large_model', 'complex_model_s', 'complex_model_m',
    'complex_model_l', 'standard_gpu', 'complex_model_m_gpu',
    'complex_model_l_gpu', 'standard_p100', 'complex_model_m_p100',
    'cloud_tpu'
]

DEFAULT_SCALE_TIER = 'BASIC_GPU'
DEFAULT_MASTER_TYPE = 'standard_gpu'
DEFAULT_WORKER_TYPE = 'standard_gpu'
DEFAULT_WORKER_COUNT = 2
DEFAULT_PS_TYPE = 'large_model'
DEFAULT_PS_COUNT = 0

DEFAULT_CONFIG_FILENAME = 'config.yml'
DEFAULT_PACKAGES_PATH = 'packages'


def check_dependencies(f):
    """
    Decorator for commands that will check if they have
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if MISSING_DEPENDENCIES:
            raise click.ClickException(
                'To use Google Cloud functionalities, you must install '
                'Luminoth with the `gcloud` extras.\n\n'
                ''
                'Ie. `pip install luminoth[gcloud]`'
            )

        return f(*args, **kwargs)

    return decorated_function


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
    tarball_path = os.path.join(output_dir, tarball_filename)

    path = upload_file(
        bucket, '{}/{}'.format(base_path, DEFAULT_PACKAGES_PATH), tarball_path
    )

    shutil.rmtree(temp_dir)

    return path


class ServiceAccount(object):
    """
    Wrapper for handling Google services via the Service Account.
    """

    def __init__(self):
        try:
            data = json.load(
                tf.gfile.GFile(
                    os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', ''), 'r'
                )
            )

            self.project_id = data['project_id']
            self.client_id = data['client_id']

            self.credentials = service_account.ServiceAccountCredentials.\
                from_json_keyfile_dict(data)
        except (ValueError, tf.errors.NotFoundError):
            click.echo(
                'Error: could not read service account credentials.\n\n'
                'Make sure the GOOGLE_APPLICATION_CREDENTIALS environment '
                'variable is set and points to a valid service account JSON '
                'file.',
                err=True
            )
            sys.exit(1)

    def cloud_service(self, service, version='v1'):
        return discovery.build(service, version, credentials=self.credentials)

    def get_bucket(self, bucket_name):
        # If not passed, will get credentials from env. This library (storage)
        # doesn't work with self.credentials.
        cli = storage.Client()
        bucket = cli.lookup_bucket(bucket_name)
        if not bucket:
            bucket = cli.create_bucket(bucket_name)
        return bucket

    def validate_region(self, region):
        regionrequest = self.cloud_service('compute').regions().get(
            region=region, project=self.project_id
        )
        try:
            regionrequest.execute()
        except errors.HttpError as err:
            if err.resp.status == 404:
                click.echo(
                    'Error: Couldn\'t find region "{}" for '
                    'project "{}".'.format(region, self.project_id),
                    err=True)
            elif err.resp.status == 403:
                click.echo('Error: Forbidden access to resources.')
                click.echo('Raw response:\n{}\n'.format(err.content))
                click.echo(
                    'Make sure to enable the following APIs for the project:\n'
                    '  * Compute Engine\n'
                    '  * Cloud Machine Learning Engine\n'
                    '  * Google Cloud Storage\n'
                    'You can do it with the following command:\n'
                    '  gcloud services enable compute.googleapis.com '
                    'ml.googleapis.com storage-component.googleapis.com\n\n'
                    'For information on how to enable these APIs, see here: '
                    'https://support.google.com/cloud/answer/6158841',
                    err=True
                )
            else:
                click.echo('Unknown error: {}'.format(err.resp), err=True)
            sys.exit(1)


def upload_data(bucket, file_path, data):
    blob = bucket.blob(file_path)
    blob.upload_from_string(data)
    return blob


def upload_file(bucket, base_path, file_path):
    filename = os.path.basename(file_path)
    path = '{}/{}'.format(base_path, filename)
    click.echo('Uploading file: "{}"\n\t-> to "gs://{}/{}"'.format(
        filename, bucket.name, path))
    blob = bucket.blob(path)
    blob.upload_from_file(tf.gfile.GFile(file_path, 'rb'))
    return path


@click.group(help='Train models in Google Cloud ML')
def gc():
    pass


@gc.command(help='Start a training job')
@click.option('--job-id', help='Identifies the training job in Google Cloud. Will use it to name the folder where checkpoints and logs will be stored, except when resuming a previous training job.')  # noqa
@click.option('--resume', 'resume_job_id', help='Id of the previous job to resume (start from last stored checkpoint). In case you are resuming multiple times, must always point to the first job (ie. the one that first created the checkpoint).')  # noqa
@click.option('--bucket', 'bucket_name', help='Bucket where to create the folder to save checkpoints and logs. If resuming a job, it must match the bucket used for the original job.')  # noqa
@click.option('--region', default='us-central1', help='Region in which to run the job.')  # noqa
@click.option('--dataset', help='Complete path (bucket included) to the folder where the dataset is located (TFRecord files).')  # noqa
@click.option('config_files', '--config', '-c', required=True, multiple=True, help='Path to config to use in training.')  # noqa
@click.option('--scale-tier', default=DEFAULT_SCALE_TIER, type=click.Choice(SCALE_TIERS))  # noqa
@click.option('--master-type', default=DEFAULT_MASTER_TYPE, type=click.Choice(MACHINE_TYPES))  # noqa
@click.option('--worker-type', default=DEFAULT_WORKER_TYPE, type=click.Choice(MACHINE_TYPES))  # noqa
@click.option('--worker-count', default=DEFAULT_WORKER_COUNT, type=int)
@click.option('--parameter-server-type', default=DEFAULT_PS_TYPE, type=click.Choice(MACHINE_TYPES))  # noqa
@click.option('--parameter-server-count', default=DEFAULT_PS_COUNT, type=int)
@check_dependencies
def train(job_id, resume_job_id, bucket_name, region, config_files, dataset,
          scale_tier, master_type, worker_type, worker_count,
          parameter_server_type, parameter_server_count):
    account = ServiceAccount()
    account.validate_region(region)

    if bucket_name is None:
        bucket_name = 'luminoth-{}'.format(account.client_id)
        click.echo(
            'Bucket name not specified. Using "{}".'.format(bucket_name))

    # Creates bucket for logs and models if it doesn't exist
    bucket = account.get_bucket(bucket_name)

    if not job_id:
        job_id = 'train_{}'.format(datetime.now().strftime("%Y%m%d_%H%M%S"))

    # Path in bucket to store job's config, logs, etc.
    # If we are resuming a previous job, then we will use the same path
    # that job used, so Luminoth will load the checkpoint from there.
    base_path = 'lumi_{}'.format(resume_job_id if resume_job_id else job_id)

    package_path = build_package(bucket, base_path)
    job_dir = 'gs://{}/{}'.format(bucket_name, base_path)

    override_params = [
        'train.job_dir={}'.format(job_dir),
    ]

    if dataset:
        # Check if absolute or relative dataset path
        if not dataset.startswith('gs://'):
            dataset = 'gs://{}'.format(dataset)
        override_params.append('dataset.dir={}'.format(dataset))

    # Even if we are resuming job, we will use a new config. Thus, we will
    # overwrite the config in the old job's dir if it existed.
    config = get_config(config_files, override_params=override_params)

    # Update final config file to job bucket
    config_path = '{}/{}'.format(base_path, DEFAULT_CONFIG_FILENAME)
    upload_data(bucket, config_path, dump_config(config))

    args = ['--config', '{}/{}'.format(job_dir, DEFAULT_CONFIG_FILENAME)]

    cloudml = account.cloud_service('ml')

    training_inputs = {
        'scaleTier': scale_tier,
        'packageUris': [
            'gs://{}/{}'.format(bucket_name, package_path)
        ],
        'pythonModule': 'luminoth.train',
        'args': args,
        'region': region,
        'jobDir': job_dir,
        'runtimeVersion': RUNTIME_VERSION
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
        body=job_spec, parent='projects/{}'.format(account.project_id))

    try:
        click.echo('Submitting training job.')
        res = jobrequest.execute()
        click.echo('Job submitted successfully.')
        click.echo('state = {}, createTime = {}'.format(
            res.get('state'), res.get('createTime')))
        if resume_job_id:
            click.echo(
                '\nNote: this job is resuming job {}.\n'.format(resume_job_id))
        click.echo('Job id: {}'.format(job_id))
        click.echo('Job directory: {}'.format(job_dir))

        save_run(config, environment='gcloud', extra_config=job_spec)

    except Exception as err:
        click.echo(
            'There was an error creating the training job. '
            'Check the details: \n{}'.format(err._get_reason())
        )


@gc.command(help='Start a evaluation job')
@click.option('--job-id', help='Job Id for naming the folder where the results of the evaluation will be stored.')  # noqa
@click.option('--train-folder', 'train_folder', required=True, help='Complete path (bucket included) where the training results are stored (config.yml should live here).')  # noqa
@click.option('--bucket', 'bucket_name', help='The bucket where the evaluation results were stored.')  # noqa
@click.option('dataset_split', '--split', default='val', help='Dataset split to use.')  # noqa
@click.option('--region', default='us-central1', help='Region in which to run the job.')  # noqa
@click.option('--machine-type', default=DEFAULT_MASTER_TYPE, type=click.Choice(MACHINE_TYPES))  # noqa
@click.option('--rebuild', default=False, is_flag=True, help='Rebuild Luminoth package for evaluation. If not, will use the same package that was used for training.')  # noqa
@check_dependencies
def evaluate(job_id, train_folder, bucket_name, dataset_split, region,
             machine_type, rebuild):
    account = ServiceAccount()
    account.validate_region(region)

    if not train_folder.startswith('gs://'):
        train_folder = 'gs://{}'.format(train_folder)

    if not job_id:
        job_id = 'eval_{}'.format(datetime.now().strftime("%Y%m%d_%H%M%S"))

    if bucket_name is None:
        bucket_name = 'luminoth-{}'.format(account.client_id)
        click.echo(
            'Bucket name not specified. Using "{}".'.format(bucket_name),
            err=True)

    if rebuild:
        job_folder = 'lumi_{}'.format(job_id)

        # Make new package in the bucket for eval results
        bucket = account.get_bucket(bucket_name)
        package_path = build_package(bucket, job_folder)
        full_package_path = 'gs://{}/{}'.format(bucket_name, package_path)
    else:
        # Get old training package from the training folder.
        # There should only be one file ending in `.tar.gz`.
        train_packages_dir = '{}/{}'.format(
            train_folder, DEFAULT_PACKAGES_PATH
        )

        try:
            package_files = tf.gfile.ListDirectory(train_packages_dir)
            package_filename = [
                n for n in package_files if n.endswith('tar.gz')
            ][0]
            full_package_path = '{}/{}'.format(
                train_packages_dir, package_filename)
        except (IndexError, tf.errors.NotFoundError):
            click.echo(
                'Could not find a `.tar.gz` Python package of Luminoth in '
                '{}.\n\nCheck that the --train-folder parameter is '
                'correct.'.format(train_packages_dir),
                err=True
            )
            sys.exit(1)

    train_config_path = '{}/{}'.format(train_folder, DEFAULT_CONFIG_FILENAME)
    cloudml = account.cloud_service('ml')

    args = [
        ['--config', train_config_path],
        ['--split', dataset_split],
    ]

    training_inputs = {
        'scaleTier': 'CUSTOM',
        'masterType': machine_type,
        'packageUris': [full_package_path],
        'pythonModule': 'luminoth.eval',
        'args': args,
        'region': region,
        # Don't need to pass jobDir since it will read it from config file.
        'runtimeVersion': RUNTIME_VERSION,
    }

    job_spec = {
        'jobId': job_id,
        'trainingInput': training_inputs
    }

    jobrequest = cloudml.projects().jobs().create(
        body=job_spec, parent='projects/{}'.format(account.project_id))

    try:
        click.echo('Submitting evaluation job.')
        res = jobrequest.execute()
        click.echo('Job submitted successfully.')
        click.echo('state = {}, createTime = {}\n'.format(
            res.get('state'), res.get('createTime')))
        click.echo('Job id: {}'.format(job_id))
        click.echo('Job directory: {}'.format('gs://{}'.format(bucket_name)))

    except Exception as err:
        click.echo(
            'There was an error creating the evaluation job. '
            'Check the details: \n{}'.format(err._get_reason()),
            err=True
        )


@gc.command(help='List project jobs')
@click.option('--running', is_flag=True, help='List only jobs that are running.')  # noqa
@check_dependencies
def jobs(running):
    account = ServiceAccount()
    cloudml = account.cloud_service('ml')
    request = cloudml.projects().jobs().list(
        parent='projects/{}'.format(account.project_id))

    try:
        response = request.execute()
        jobs = response['jobs']

        if not jobs:
            click.echo('There are no jobs for this project.')
            return

        if running:
            jobs = [j for j in jobs if j['state'] == 'RUNNING']
            if not jobs:
                click.echo('There are no running jobs.')
                return

        for job in jobs:
            click.echo('Id: {} Created: {} State: {}'.format(
                job['jobId'], job['createTime'], job['state']))
    except Exception as err:
        click.echo(
            'There was an error fetching jobs. '
            'Check the details: \n{}'.format(err._get_reason()),
            err=True
        )


@gc.command(help='Show logs from a running job')
@click.option('--job-id', required=True)
@click.option('--polling-interval', default=30, help='Polling interval in seconds.')  # noqa
@check_dependencies
def logs(job_id, polling_interval):
    account = ServiceAccount()
    cloudlog = account.cloud_service('logging', 'v2')

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
                'resourceNames': 'projects/{}'.format(account.project_id),
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
                    'Check the details: \n{}'.format(err._get_reason()),
                    err=True
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
