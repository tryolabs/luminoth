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
Luminoth can easily run in `Google Cloud ML Engine <https://cloud.google.com/ml-engine/>`_
with a single command.

For more information, see :ref:`usage/cloud`.
