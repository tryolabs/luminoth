# Training

## Locally

You can use the `lumi train <model-type>` command to easily start training your model (with default config). Before anything else, try running it to see if everything is working.

The `train` cli tool provides many options related to training.

Options:
  - `--config`/`-c`: Config file to use.
  - `--override`/`-o`: Override any configuration setting using dot notation (e.g.: `-o rpn.proposals.nms_threshold=0.8`).
  - `--run-name`: Run name used for logs and checkpoints.
  - `--continue-training`: Automatically search for checkpoints and continue training from the last one.
  - `--model-dir`: Where to save the training checkpoints.
  - `--checkpoint-file`: From where to read network weights (if available).
  - `--log-dir`: Where to save the training logs and summaries.
  - `--save-every`: Save a checkpoint every that many batches or seconds.
  - `--debug`: Debug log level and richer tensor outputs.
  - `--tf-debug`: TensorFlow's tfdb debugger.
  - `--save-timeline`: Save timeline of execution.
  - `--full-trace`: Run TensorFlow with `FULL_TRACE` config for memory and running time debugging in TensorBoard.


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
  - `--dataset`: Path to dataset in the bucket provided.
  - `--scale-tier`: Cluster configuration. Default: BASIC_GPU.
  - `--master-type`: Master node machine type.
  - `--worker-type`: Worker node machine type.
  - `--worker-count`: Number of workers.

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
