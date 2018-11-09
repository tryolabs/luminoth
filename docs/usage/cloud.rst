.. usage/cloud:

Cloud management
================

We support training in `Google Cloud ML engine <https://cloud.google.com/ml-engine/>`_,
which has native Tensorflow support.

Instead of building Python packages yourself and using Google Cloud SDK, we baked
the process inside Luminoth itself, so you can pull it of with a few simple commands.

You can choose how many workers you want, which `scale tiers
<https://cloud.google.com/ml-engine/docs/concepts/training-overview#scale_tier>`_
to use, and where to store the results. We also provide some utilities to
monitor and manage your job right from your command line.

For all the cloud functionalities, the files read (such as the datasets in
TFRecord format) and written (such as logs and checkpoints) by Luminoth will reside
in `Buckets <https://cloud.google.com/storage/docs/creating-buckets>`_ instead of your
local disk.

Pre-requisites
``````````````

#. Create a `Google Cloud project <https://console.cloud.google.com/projectcreate>`_.
#. Install `Google Cloud SDK <https://cloud.google.com/sdk/>`_ on your machine.
   Although it is not strictly necessary, it will be useful to enable the required
   APIs and upload your dataset.
#. Login with Google Cloud SDK::

    gcloud auth login

#. Enable the following APIs:
     * Compute Engine
     * Cloud Machine Learning Engine
     * Google Cloud Storage

   You can do it through the `web console <https://support.google.com/cloud/answer/6158841>`_
   or with the following command::

     gcloud services enable compute.googleapis.com ml.googleapis.com storage-component.googleapis.com

   Be patient, it can take a few minutes!

#. Upload your dataset's TFRecord files to a Cloud Storage bucket::

    gsutil -o GSUtil:parallel_composite_upload_threshold=150M cp -r /path/to/dataset/tfrecords gs://your_bucket/path

#. Create a `Service Account Key <https://console.cloud.google.com/iam-admin/serviceaccounts/project>`_
   (JSON format) and download it to your directory of choice. You may add it as an Editor
   of your project. If necessary, add required roles (permissions) to your service
   account.

#. Point the ``GOOGLE_APPLICATION_CREDENTIALS`` environment variable to the JSON file of
   the service account.

Running a training job
``````````````````````

Luminoth command line tool provides commands to submit training jobs, list them
and fetch their logs.

``lumi cloud gc train`` - Submit a training job.

Required arguments:
  - ``--config``: YAML configuration file for use in training.

Optional arguments:
  - ``--dataset``: full path to bucket with the dataset's TFRecord files, ie.
    ``gs://<bucket_name>/<path>``. If not present, will default from the
    value specified in the YAML config file (``dataset.dir``).
  - ``--resume``: Id of the previous job to resume (start from last stored checkpoint). In
    case you are resuming multiple times, must always point to the first job (ie. the one
    that first created the checkpoint). - ``--bucket``: Bucket name for storing data for
    the job, such as the logs.
    Defaults to ``luminoth-<client_id>``.
  - ``--job-id``: Identifies the training job in Google Cloud. Defaults to
    ``train_<timestamp>``.
  - ``--region``: `Google Cloud region
    <https://cloud.google.com/compute/docs/regions-zones/>`_ in which to set up
    the cluster.
  - ``--scale-tier``: Cluster configuration. Default: ``BASIC_GPU``.
  - ``--master-type``: Master node machine type.
  - ``--worker-type``: Worker node machine type.
  - ``--worker-count``: Number of workers.
  - ``--parameter-server-type``: Parameter server node machine type.
  - ``--parameter-server-count``: Number of parameter servers.

Example::

    lumi cloud gc train \
        --bucket luminoth-train-jobs \
        --dataset gs://luminoth-train-datasets/coco/tfrecords \
        -c config.yml

Resuming a previous training job
````````````````````````````````
Sometimes, you may wish to restart a previous training job without losing all
the progress made so far (ie. resume from checkpoint). For example, it might be
the case that you have updated your TFRecords dataset and want your model
fine-tuned with the new data.

The way to achieve this in Google Cloud is by launching a **new training job**,
but telling Luminoth to resume a previous job id::

    lumi cloud gc train \
        --resume <previous-job-id> \
        --bucket luminoth-train-jobs \
        --dataset gs://luminoth-train-datasets/coco/tfrecords \
        -c config.yml

Keep in mind that for this to work:
  - ``bucket`` must match the same bucket name that was used for the
    job you are resuming.
  - In case you are resuming a job multiple times, ``previous-job-id`` must
    be the id of the job that first created the checkpoint. This is so
    Luminoth keeps writing the new files to the same folder.


Listing jobs
````````````

``lumi cloud gc jobs`` - List project's jobs.

Optional arguments:
  - ``--running``: Show running jobs only.

Fetching logs
`````````````

``lumi cloud gc logs`` - Fetch logs for a specific job.

Required arguments:
  - ``--job-id``: id of the training job, obtained after running ``lumi cloud gc train``.

Optional arguments:
  - ``--polling-interval``: Seconds between each log request.

Running an evaluation job
`````````````````````````

``lumi cloud gc evaluate`` - Submit an evaluation job.

Required arguments:
  - ``--train-folder``: Complete path (bucket included) where the training results
    are stored (config.yml should live here).

Optional arguments:
  - ``--split``: Dataset split to use. Defaults to ``val``.
  - ``--job-id``: Job Id for naming the folder where the results of the evaluation will be stored.
  - ``--bucket``: The bucket where the evaluation results were stored.
  - ``--region``: `Google Cloud region
    <https://cloud.google.com/compute/docs/regions-zones/>`_ in which to run the job.
  - ``--scale-tier``: Cluster configuration. Default: ``BASIC_GPU``.
  - ``--master-type``: Master node machine type.
  - ``--rebuild``: Whether to rebuild the package with the currently installed version of Luminoth,
    or use the same Luminoth package that was used for training.

Example::

    lumi cloud gc evaluate \
        --train-folder gs://luminoth-train-jobs/lumi_train_XXXXXXXX_YYYYYY \
        --bucket luminoth-eval-jobs \
        --split test

Results
```````

Everything related to a job is stored in its own folder on the bucket provided
under the name ``lumi_{job_id}``.
