.. _tutorial/03-training-the-model:

Training the model
==================

Now that we have created our (toy) dataset, we can proceed to train our model.

The configuration file
----------------------

Training orchestration, including the model to be used, the dataset location and training
schedule, is specified in a YAML config file. This file will be consumed by Luminoth and
merged to the default configuration, to start the training session.

You can see a minimal config file example in
`sample_config.yml <https://github.com/tryolabs/luminoth/blob/master/examples/sample_config.yml>`_.
This file illustrates the entries you'll most probably need to modify, which are:

* ``train.run_name``: the run name for the training session, used to identify it.
* ``train.job_dir``: directory in which both model checkpoints and summaries (for
  TensorBoard consumption) will be saved. The actual files will be stored under
  ``<job_dir>/<run_name>``.
* ``dataset.dir``: directory from which to read the TFRecord files.
* ``model.type``: model to use for object detection (``fasterrcnn``, or ``ssd``).
* ``network.num_classes``: number of classes to predict (depends on your dataset).

For looking at all the possible configuration options,  mostly related to the model
itself, you can check the
`base_config.yml <https://github.com/tryolabs/luminoth/blob/master/luminoth/models/fasterrcnn/base_config.yml>`_
file.

Building the config file for your dataset
-----------------------------------------

Probably the most important setting for training is the **learning rate**. You will most
likely want to tune this depending on your dataset, and you can do it via the
``train.learning_rate`` setting in the configuration. For example, this would be a good
setting for training on the full COCO dataset:

.. code-block:: yaml

   learning_rate:
     decay_method: piecewise_constant
     boundaries: [250000, 450000, 600000]
     values: [0.0003, 0.0001, 0.00003, 0.00001]

To get to this, you will need to run some experiments and see what works best.

.. code-block:: yaml

   train:
     # Run name for the training session.
     run_name: traffic
     job_dir: <change this directory>
     learning_rate:
       decay_method: piecewise_constant
       # Custom dataset for Luminoth Tutorial
       boundaries: [90000, 160000, 250000]
       values: [0.0003, 0.0001, 0.00003, 0.00001]
   dataset:
     type: object_detection
     dir: <directory with your dataset>
   model:
     type: fasterrcnn
     network:
       num_classes: 8
     anchors:
       # Add one more scale to be better at detecting small objects
       scales: [0.125, 0.25, 0.5, 1, 2]

Running the training
--------------------

Assuming you already have both your dataset (TFRecords) and the config file ready, you can
start your training session by running the command as follows:

.. code-block:: bash

   lumi train -c config.yml

You can use the ``-o`` option to override any configuration option using dot notation (e.g.
``-o model.rpn.proposals.nms_threshold=0.8``).

If you are using a CUDA-based GPU, you can select the GPU to use by setting the
``CUDA_VISIBLE_DEVICES`` environment variable (see
`here <https://devblogs.nvidia.com/cuda-pro-tip-control-gpu-visibility-cuda_visible_devices/>`_
for more info).

When the training is running, you should see Luminoth print out for each step, the
minibatch (single image), and the training loss related to that minibatch.

Image to image, the training loss will jump around, and this is expected. However, the
trend will be that the loss will gradually start to decrease. For this, it is interesting
to look at it using tools like TensorBoard.

Storing partial weights (checkpoints)
-------------------------------------

As the training progresses, Luminoth will periodically save a checkpoint with the current
weights of the model. These weights let you resume training from where you left off!

The files will be output in your ``<job_dir>/<run_name>`` folder. By default, they will be
saved every 600 seconds of training, but you can configure this with the
``train.save_checkpoint_secs`` setting in your config file.

The default is to only store the latest checkpoint (that is, when a checkpoint is
generated, the previous checkpoint gets deleted) in order to conserve storage. You might
find the ``train.checkpoints_max_keep`` option in your train YML configuration useful if
you want to keep more checkpoints around.

----

Next: :ref:`tutorial/04-visualizing-the-training-process`
