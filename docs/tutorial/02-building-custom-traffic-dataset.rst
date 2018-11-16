.. _tutorial/02-building-custom-traffic-dataset:

Building custom traffic dataset
===============================

Even though pre-trained checkpoints are really useful, most of the time you will want to
train an object detector using your own dataset. For this, you need a source of images and
their corresponding bounding box coordinates and labels, in some format that Luminoth can
understand. In this case, we are interested in street traffic related objects, so we will
need to source images relevant to our niche.

How Luminoth handles datasets
-----------------------------

Luminoth reads datasets natively only in TensorFlow's `TFRecords
format <https://www.tensorflow.org/guide/datasets#consuming_tfrecord_data>`_. This is a
binary format that will let Luminoth consume the data very efficiently.

In order to use a custom dataset, you must first transform whatever format your data is
in, to TFRecords files (one for each split — train, val, test). Fortunately, Luminoth
provides several CLI tools (see :ref:`usage/dataset`) for transforming popular dataset
format (such as Pascal VOC, ImageNet, COCO, CSV, etc.) into TFRecords. In what follows, we
will leverage this.

Building a traffic dataset using OpenImages
-------------------------------------------

`OpenImages V4 <https://storage.googleapis.com/openimages/web/index.html>`__ is the
largest existing dataset with object location annotations. It contains 15.4M
bounding-boxes for 600 categories on 1.9M images, making it a very good choice for getting
example images of a variety of (not niche-domain) classes (persons, cars, dolphin,
blender, etc).

Preparing the data
^^^^^^^^^^^^^^^^^^

We should start by downloading `the annotation
files <https://storage.googleapis.com/openimages/web/download.html>`__
(`this <https://storage.googleapis.com/openimages/2018_04/train/train-annotations-bbox.csv>`__
and `this <https://storage.googleapis.com/openimages/2018_04/train/train-annotations-human-imagelabels-boxable.csv>`__,
for train) and the `class description <https://storage.googleapis.com/openimages/2018_04/class-descriptions-boxable.csv>`_
file. Note that the files with the annotations themselves are pretty large, totalling over
1.5 GB (and this CSV files only, without downloading a single image!).

After we get the ``class-descriptions-boxable.csv`` file, we can go over all the classes
available in the OpenImages dataset and see which ones are related to **traffic**. The
following were hand-picked after examining the full file:

.. code-block:: text

   /m/015qff,Traffic light
   /m/0199g,Bicycle
   /m/01bjv,Bus
   /m/01g317,Person
   /m/04_sv,Motorcycle
   /m/07r04,Truck
   /m/0h2r6,Van
   /m/0k4j,Car

Using the Luminoth dataset reader
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Luminoth includes a **dataset reader** that can take OpenImages format. As the dataset is
so large, this will never download every single image, but fetch only those we want to use
and store them directly in the TFRecords file.

Note that the dataset reader expects a particular directory layout so it knows where the
files are located. In this case, files corresponding to the examples must be in a folder
named like their split (`train`, `test`, ...). So, you should have the following:

.. code-block:: text

    .
    ├── class-descriptions-boxable.csv
    └── train
        ├── train-annotations-bbox.csv
        └── train-annotations-human-imagelabels-boxable.csv

Next, run the following command:

.. code-block:: bash

   lumi dataset transform \
         --type openimages \
         --data-dir . \
         --output-dir ./out \
         --split train  \
         --class-examples 100 \
         --only-classes=/m/015qff,/m/0199g,/m/01bjv,/m/01g317,/m/04_sv,/m/07r04,/m/0h2r6,/m/0k4j

This will generate TFRecord file for the ``train`` split. You should get something like
this in your terminal after the command finishes:

.. code-block:: text

   INFO:tensorflow:Saved 360 records to "./out/train.tfrecords"
   INFO:tensorflow:Composition per class (train):
   INFO:tensorflow:        Person (/m/01g317): 380
   INFO:tensorflow:        Car (/m/0k4j): 255
   INFO:tensorflow:        Bicycle (/m/0199g): 126
   INFO:tensorflow:        Bus (/m/01bjv): 106
   INFO:tensorflow:        Traffic light (/m/015qff): 105
   INFO:tensorflow:        Truck (/m/07r04): 101
   INFO:tensorflow:        Van (/m/0h2r6): 100
   INFO:tensorflow:        Motorcycle (/m/04_sv): 100


Apart from the TFRecord file, you will also get a ``classes.json`` file that lists the names
of the classes in your dataset.

Note that:

* As we are using ``--only-classes``, so we filter to only use the classes we care about.
* We are using ``--max-per-class`` of 100. This setting will make it stop when every class
  has at least 100 examples. However, some classes may end up with many more; for example
  here it needed to get 380 instances of persons to get 100 motorcycles, considering the
  first 360 images.
* We could also have used ``--limit-examples`` option so we know the number of records in
  our final dataset beforehand.

Of course, this dataset is **way too small** for any meaningful training to go on, but we
are just showcasing. In real life, you would use a much larger value for
``--max-per-class`` (ie. 15000) or ``--limit-examples``.

----

Next: :ref:`tutorial/03-training-the-model`
