.. _usage/installation:

Installation
============

Before you start
----------------

TensorFlow
^^^^^^^^^^

To use Luminoth, `TensorFlow <https://tensorflow.org>`_ must be installed beforehand.

If you want **GPU support**, you should install the GPU version of TensorFlow with
``pip install tensorflow-gpu``, or else you can use the CPU version using
``pip install tensorflow``.

You can see more details of how to install TensorFlow manually `here
<https://www.tensorflow.org/install/>`__, including how to use CUDA and cuDNN.

FFmpeg
^^^^^^

Luminoth leverages `FFmpeg <https://www.ffmpeg.org>`_ in order to support
running predictions on videos. If you plan to use Luminoth with this end,
FFmpeg should be installed as a system dependency.


Installing from PyPI
--------------------

Use ``pip`` to install Luminoth, by running the following command::

  pip install luminoth

Google Cloud
^^^^^^^^^^^^

If you wish to train using **Google Cloud ML Engine**, the optional dependencies
must be installed::

  $ pip install luminoth[gcloud]


Installing from source
----------------------

Start by cloning the Luminoth repository::

  git clone https://github.com/tryolabs/luminoth.git

Then install the library by running::

  cd luminoth
  pip install -e .
