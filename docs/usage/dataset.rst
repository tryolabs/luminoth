.. _usage/dataset:

Adapting a dataset
==================

If a pre-trained model for the task you want to perform is not available, you
can train Luminoth with an existing open dataset, or even your own.

The first step in training Luminoth to convert your dataset to TensorFlow's
``.tfrecords`` format. This ensures that no matter what image or annotation
formats the original dataset uses, it will be transformed to something that
Luminoth can understand and process efficiently, either while training locally
or in the cloud.

For this purpose, Luminoth provides a conversion tool which includes support for
some of the most well-known datasets for object detection and classification
tasks.

Conversion tool
---------------

As explained above, the conversion tool, invoked with the command ``lumi dataset
transform``, allows you to transform a dataset in a standard format into one
that can be understood by Luminoth.

Supported datasets
^^^^^^^^^^^^^^^^^^

You can select the annotation scheme used by your dataset with the ``--type``
option. As long as your dataset follows the same scheme, the conversion tool
will be able to transform it correctly. Furthermore, you can write your own
conversion tool to read a custom format (see :ref:`custom-conversion`).

The supported types are:

- ``pascal``: format used by the `Pascal VOC
  <http://host.robots.ox.ac.uk:8080/pascal/VOC/voc2012/index.html>`_ dataset.

- ``imagenet``: format used by the `ImageNet <http://image-net.org/download>`_
  dataset.

- ``coco``: format used by the `COCO <http://cocodataset.org/#download>`_
  dataset.

Input and output
^^^^^^^^^^^^^^^^

In order to point the conversion tool to the actual data to transform, you must
set the ``--data-dir`` option to the directory containing it. This path should
follow the directory structure expected by the indicated ``--type``. For
instance, in the case of the ``pascal`` dataset type, this will be the
``VOCdevkit/VOC2007`` directory obtained from extracting the tar file provided
by the dataset page.

The output directory is specified with the ``--output-dir`` option. This is the
path where the TFrecords files will be stored, so make sure there's enough space
in the disk.

You can also specify which dataset splits (i.e. train, validation or test) to
convert, whenever that information is available. You can do so by using the
``--split <train|val|test>`` option, using it more than once if you want to
transform more than one split at the same time.

Limiting the dataset
^^^^^^^^^^^^^^^^^^^^

For datasets with many classes you might want to ignore certain classes when
training a custom detector. For instance, if you want to train a traffic
detector, you could start with the COCO dataset but only use, out of the eighty
classes present in it, cars, trucks, buses and motorcycles. You can do so with
the ``--only-classes`` option, by passing a comma-separated list of classes to
keep in the final dataset.

During development, it is often useful to verify that the model can actually
overfit a small dataset. You can use the ``--limit-examples`` and
``--limit-classes`` options for this, allowing you to create a dataset limited
to up to ``N`` examples and/or ``M`` random classes.

Examples
^^^^^^^^

Say we want to transform the ``train`` and ``val`` splits of the Pascal VOC2012
dataset.  This will output the corresponding ``.tfrecords`` files to the output
dir::

  $ lumi dataset transform \
          --type pascal \
          --data-dir datasets/pascal/VOCdevkit/VOC2012/ \
          --output-dir datasets/pascal/tf/ \
          --split train --split val

If we wanted to use COCO to create a traffic-specific dataset, we could use the
following command::

  $ lumi dataset transform \
          --type coco \
          --data-dir datasets/coco/ \
          --output-dir datasets/coco/tf/ \
          --split train --split val
          --only-classes=car,truck,bus,motorcycle,bicycle

.. _custom-conversion:

Supporting your own dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^

TODO: Guidelines on how to write your own dataset reader.

For now, you can see ``luminoth/tools/dataset/readers/object_detection/pascalvoc.py``
as an example on creating your own reader.

Merge tool
----------

Sometimes you don't have a dataset for your model, but are able to leverage data
from several open datasets. Luminoth provides a dataset merging tool for this
purpose, allowing you to combine several TFrecords files (i.e. already converted
into Luminoth's expected format) into a single one.

This tool is provided through the ``lumi dataset merge`` command, which receives
a list of TFrecords files and outputs it to the file indicated by the last
argument. For example::

  $ lumi dataset merge \
          datasets/pascal/tf/2007/only-traffic/train.tfrecords \
          datasets/pascal/tf/2012/only-traffic/train.tfrecords \
          datasets/coco/tf/only-traffic/train.tfrecords \
          datasets/tf/train.tfrecords
