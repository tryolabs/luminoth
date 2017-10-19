# Training

## Locally

You can use the `lumi train -c sample_config.yml` command to easily start training your model (with default config). Before anything else, try running it to see if everything is working. You'll need a dataset in tfrecord format in the default location (`./datasets/voc/tf`). See [DATASETS.md](./DATASETS.md) for more info.

The `train` cli tool provides the following options related to training.

Options:
  - `--config`/`-c`: Config file to use. If the flag is repeated, all config files will be merged in left-to-right order so that every file overwrites the configuration of keys defined previously.
  - `--override`/`-o`: Override any configuration setting using dot notation (e.g.: `-o model.rpn.proposals.nms_threshold=0.8`).

Most of the configuration is done via the `--config` file. See the
[sample_config.yml](/sample_config.yml) for a simple example, or fasterrcnn's
[base_config.yml](/luminoth/models/fasterrcnn/base_config.yml) for the full
range of settings.


## Google Cloud

We support training on Google’s Cloud ML Engine, which has native TensorFlow support. Instead of making you run a bunch of commands with lots of options, we streamlined the process and developed a simple but effective utility to easily run Luminoth.

You can choose how many workers you want, which [scale tiers](https://cloud.google.com/ml-engine/docs/concepts/training-overview#scale_tier) to use, and where to store the results. We also provide some utilities to monitor and manage your job right from your command line.

### Pre-requisites

1. Create a [Google Cloud project](https://console.cloud.google.com/projectcreate).
2. Install [Google Cloud SDK](https://cloud.google.com/sdk/) on your machine.
3. Manual login:
```
$ gcloud auth login
```
4. Your dataset needs to be available for Google Cloud ML resources. To upload it run:
```
$ gsutil -o GSUtil:parallel_composite_upload_threshold=150M cp -r /path/to/dataset/tfrecords gs://your_bucket/path
```
5. Create a [Service Account Key](https://console.cloud.google.com/iam-admin/serviceaccounts/project) (JSON format) and download it to your directory of choice.

### Train

Luminoth command line tool provides commands to submit training jobs, list them and fetch their logs.

The following options are required by *all* `lumi cloud gc` sub-commands:

  - `--project-id`: Id of the project created in step 1.
  - `--service-account-json`: Path to the Service Account Key file created in step 5.

#### `lumi cloud gc train`
Submit a training job.

Options:
  - `--job-id`: Identifies the training job.
  - `--config`: Configuration used in training.
  - `--bucket`: Google Storage bucket name.
  - `--region`: [Google Cloud
region](https://cloud.google.com/compute/docs/regions-zones/) in which to set
up the cluster.
  - `--dataset`: Path to dataset in the bucket provided.
  - `--scale-tier`: Cluster configuration. Default: BASIC_GPU.
  - `--master-type`: Master node machine type.
  - `--worker-type`: Worker node machine type.
  - `--worker-count`: Number of workers.
  - `--parameter-server-type`: Parameter server node machine type.
  - `--parameter-server-count`: Number of parameter servers.

#### `lumi cloud gc jobs`
List project’s jobs.

Options:
  - `--running`: Show running jobs only.

#### `lumi cloud gc logs`
Fetch logs for a specific job.

Options:
  - `--job-id`
  - `--polling-interval`: Seconds between each log request.

### Results

Everything related to a job is stored on its own folder on the bucket provided under the name `lumi_{job_id}`. This folder has the following structure:

`lumi_{job_id}/`
  - `logs/`: Directory for Tensorboard logs.
  - `model/`: Directory to save the partial trained models.
