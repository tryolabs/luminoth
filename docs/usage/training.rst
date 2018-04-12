.. _usage/training:

Training your own model
=======================

In order to train your own model, two things are required:

* A dataset ready to be consumed by Luminoth (see :ref:`usage/dataset`).
* A configuration file for the run.

We'll start by covering the configuration file, then proceed to the training
itself, both locally and in the cloud.

Configuration
-------------

Training orchestration, including the model to be used, the dataset location
and training schedule, is specified in a YAML config file. This file will be
consumed by Luminoth and merged to the default configuration to start the
training session.

You can see a minimal config file example in `sample_config.yml
<https://github.com/tryolabs/luminoth/tree/master/examples/sample_config.yml>`_.
This file illustrates the entries you'll most probably need to modify, which
are:

* ``train.run_name``: The run name for the training session, used to identify
  it.
* ``train.job_dir``: Directory in which both model checkpoints and summaries
  (for Tensorboard consumption) will be saved. The actual files will be stored
  under ``{job_dir}/{run_name}``, so serving ``{job_dir}`` with Tensorboard will
  allow you to see all your runs at once.
* ``dataset.dir``: Directory from which to read the TFrecords files.
* ``model.type``: Model to use for object detection (e.g. ``fasterrcnn``,
  ``ssd``).
* ``network.num_classes``: Number of classes to predict.

There are a great deal of configuration options, mostly related to the model
itself. You can, for instance, see the full range of options for the Faster
R-CNN model, along with a brief description of each, in its `base_config.yml
<https://github.com/tryolabs/luminoth/tree/master/luminoth/models/fasterrcnn/base_config.yml>`_
file.

Training
--------

The model training itself can either be run locally (on the CPU or GPU
available) or in Google Cloud's Cloud ML Engine.

Locally
^^^^^^^

Assuming you already have both your dataset and the config file ready, you can
start your training session by running the command as follows::

  $ lumi train -c my_config.yml

The ``lumi train`` CLI tool provides the following options related to training.

* ``--config``/``-c``: Config file to use. If the flag is repeated, all config
  files will be merged in left-to-right order so that every file overwrites the
  configuration of keys defined previously.

* ``--override``/``-o``: Override any configuration setting using dot notation
  (e.g.: ``-o model.rpn.proposals.nms_threshold=0.8``).

If you're using a CUDA-based GPU, you can select the GPU to use by setting the
``CUDA_VISIBLE_DEVICES`` environment variable. (See the `NVIDIA site
<https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#env-vars>`_
for more information.)

You can run `Tensorboard
<https://www.tensorflow.org/programmers_guide/summaries_and_tensorboard>`_ on
the ``job_dir`` to visualize training, including the loss, evaluation metrics,
training speed, and even partial images.

Google Cloud
^^^^^^^^^^^^

We support training in Google's Cloud ML Engine, which has native Tensorflow
support. Instead of making you run a bunch of commands with lots of options, we
streamlined the process and developed a simple but effective utility to easily
run Luminoth.

You can choose how many workers you want, which `scale tiers
<https://cloud.google.com/ml-engine/docs/concepts/training-overview#scale_tier>`_
to use, and where to store the results. We also provide some utilities to
monitor and manage your job right from your command line.

Pre-requisites
``````````````

#. Create a `Google Cloud project <https://console.cloud.google.com/projectcreate>`_.
#. Install `Google Cloud SDK <https://cloud.google.com/sdk/>`_ on your machine.
#. Manual login::

    $ gcloud auth login

#. Your dataset needs to be available for Google Cloud ML resources. To upload it run::

    $ gsutil -o GSUtil:parallel_composite_upload_threshold=150M cp -r /path/to/dataset/tfrecords gs://your_bucket/path

#. Create a `Service Account Key <https://console.cloud.google.com/iam-admin/serviceaccounts/project>`_
   (JSON format) and download it to your directory of choice.

Train
`````

Luminoth command line tool provides commands to submit training jobs, list them
and fetch their logs.

The following options are required by *all* ``lumi cloud gc`` sub-commands:

* ``--project-id``: Id of the project created in step 1.
* ``--service-account-json``: Path to the Service Account Key file created in
  step 5.

``lumi cloud gc train`` - Submit a training job.

Options:
  - ``--job-id``: Identifies the training job.
  - ``--config``: Configuration used in training.
  - ``--bucket``: Google Storage bucket name.
  - ``--region``: `Google Cloud region
    <https://cloud.google.com/compute/docs/regions-zones/>`_ in which to set up
    the cluster.
  - ``--dataset``: Path to dataset in the bucket provided.
  - ``--scale-tier``: Cluster configuration. Default: BASIC_GPU.
  - ``--master-type``: Master node machine type.
  - ``--worker-type``: Worker node machine type.
  - ``--worker-count``: Number of workers.
  - ``--parameter-server-type``: Parameter server node machine type.
  - ``--parameter-server-count``: Number of parameter servers.

``lumi cloud gc jobs`` - List project’s jobs.

Options:
  - ``--running``: Show running jobs only.

``lumi cloud gc logs`` - Fetch logs for a specific job.

Options:
  - ``--job-id``
  - ``--polling-interval``: Seconds between each log request.

Results
```````

Everything related to a job is stored in its own folder on the bucket provided
under the name ``lumi_{job_id}``. This folder has the following structure:

``lumi_{job_id}/``

  * ``logs/``: Directory for Tensorboard logs.
  * ``model/``: Directory to save the partial trained models.
