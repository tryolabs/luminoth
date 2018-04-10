.. _usage/installation:

Installation
============

Before you start
----------------

Tensorflow
^^^^^^^^^^

All Luminoth dependencies will be downloaded automatically when you install
Luminoth. However, `Tensorflow <https://tensorflow.org>`_, upon which Luminoth
depends, provides two packages in PyPI: ``tensorflow`` (the CPU version) and
``tensorflow-gpu`` (the GPU version).

Luminoth will default to the CPU version of the dependency. So, if you have a
GPU, you should manually pre-install ``tensorflow-gpu`` before installing
Luminoth by issuing::

  $ pip install tensorflow-gpu

You can see more details into installing Tensorflow manually `here
<https://www.tensorflow.org/install/>`_, including how to use CUDA and cuDNN.

FFmpeg
^^^^^^

Luminoth leverages `FFmpeg <https://www.ffmpeg.org>`_ in order to support
running predictions on videos. If you plan to use Luminoth with this end,
FFmpeg should be installed as a system dependency.

Installing from PyPI
--------------------

Use ``pip`` to install Luminoth, by running the following command::

  $ pip install luminoth

Installing from source
----------------------

Start by cloning the Luminoth repository::

  $ git clone https://github.com/tryolabs/luminoth.git

Then install the library by running::

  $ cd luminoth
  $ python setup.py install
