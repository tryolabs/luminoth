.. _usage/quickstart:

Getting started
===============

After going through the installation process (see :ref:`usage/installation`),
the ``lumi`` CLI tool should be at your disposal. This tool is the main way to
interact with Luminoth, allowing you to train new models, evaluate them, use
them for predictions, manage your checkpoints and more. Running it will provide
additional information::

  Usage: lumi [OPTIONS] COMMAND [ARGS]...

  Options:
    -h, --help  Show this message and exit.

  Commands:
    checkpoint  Groups of commands to manage checkpoints
    cloud       Groups of commands to train models in the...
    dataset     Groups of commands to manage datasets
    eval        Evaluate trained (or training) models
    predict     Obtain a model's predictions.
    server      Groups of commands to serve models
    train       Train models

We'll start by downloading a checkpoint. Luminoth provides already-trained
models so you can run predictions and get reasonable results in no time (and
eventually be able to use them for fine-tuning). In order to access these
checkpoints, we first need to download the remote index with the available
models.

Checkpoint management is handled by the ``lumi checkpoint`` subcommand. Run the
following to both retrieve and list the existing checkpoints::

  $ lumi checkpoint refresh
  Retrieving remote index... done.
  2 new remote checkpoints added.
  $ lumi checkpoint list
  ================================================================================
  |           id |                  name |       alias | source |         status |
  ================================================================================
  | 48ed2350f5b2 |   Faster R-CNN w/COCO |    accurate | remote | NOT_DOWNLOADED |
  | e3256ffb7e29 |      SSD w/Pascal VOC |        fast |  local | NOT_DOWNLOADED |
  ================================================================================

Two checkpoints are present:

- **Faster R-CNN w/COCO** (48ed2350f5b2): object detection model trained on the
  Faster R-CNN model using the COCO dataset. Aliased as ``accurate``, as it's
  the slower but more accurate detection model.

- **SSD w/Pascal VOC** (e3256ffb7e29): object detection model trained on the
  Single Shot Multibox Detector (SSD) model using the Pascal dataset. Aliased
  as ``fast``, as it's the faster but less accurate detection model.

Additional commands are available for managing checkpoints, including inspection
and modification of checkpoints (see :ref:`cli/checkpoint`).  For now, we'll
download a checkpoint and use it::

  $ lumi checkpoint download 48ed2350f5b2
  Downloading checkpoint...  [####################################]  100%
  Importing checkpoint... done.
  Checkpoint imported successfully.

Once the checkpoint is downloaded, it can be used for predictions. There are
currently two ways to do this:

- Using the CLI tool and passing it either images or videos. This will output a
  JSON with the results and optionally draw the bounding boxes of the
  detections in the image.
- Using the web app provided for testing purposes. This will start a web server
  that, when connected, allows you to upload the image. Also useful to run on
  a remote GPU. (Note, however, that using Luminoth through the web interface is
  **not** production-ready and will not scale.)

Let's start with the first, by running it on an image aptly named
``image.png``::

  $ lumi predict image.png
  Found 1 files to predict.
  Neither checkpoint not config specified, assuming `accurate`.
  Predicting image.jpg... done.
  {
    "file": "image.jpg",
    "objects": [
      {"bbox": [294, 231, 468, 536], "label": "person", "prob": 0.9997},
      {"bbox": [494, 289, 578, 439], "label": "person", "prob": 0.9971},
      {"bbox": [727, 303, 800, 465], "label": "person", "prob": 0.997},
      {"bbox": [555, 315, 652, 560], "label": "person", "prob": 0.9965},
      {"bbox": [569, 425, 636, 600], "label": "bicycle", "prob": 0.9934},
      {"bbox": [326, 410, 426, 582], "label": "bicycle", "prob": 0.9933},
      {"bbox": [744, 380, 784, 482], "label": "bicycle", "prob": 0.9334},
      {"bbox": [506, 360, 565, 480], "label": "bicycle", "prob": 0.8724},
      {"bbox": [848, 319, 858, 342], "label": "person", "prob": 0.8142},
      {"bbox": [534, 298, 633, 473], "label": "person", "prob": 0.4089}
    ]
  }

You can further specify the checkpoint to use (by using the ``--checkpoint``
option), as well as indicating the minimum score to allow for bounding boxes
(too low will detect noise, too high and won't detect anything), the number of
detections, and so on.

The second variant is even easier to use, just run the following command and go
to `<http://127.0.0.1:5000>`_::

  $ lumi server web
  Neither checkpoint not config specified, assuming `accurate`.
   * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)

In there, you'll be able to upload an image and see the results.

And that's it for the basics! Next steps would be:

- Prepare your own dataset to be consumed by Luminoth (see :ref:`usage/dataset`).
- Train a custom model with your own data, either locally or in Google Cloud
  (see :ref:`usage/training`).
- Turn your custom model into a checkpoint for easier sharing and usage (see
  :ref:`usage/checkpoints`).
- Use the Python API to call Luminoth models within Python.
