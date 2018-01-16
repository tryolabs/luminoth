[![Luminoth](https://user-images.githubusercontent.com/270983/31414425-c12314d2-ae15-11e7-8cc9-42d330b03310.png)](https://luminoth.ai)
========

[![Build Status](https://travis-ci.org/tryolabs/luminoth.svg?branch=master)](https://travis-ci.org/tryolabs/luminoth)
[![codecov](https://codecov.io/gh/tryolabs/luminoth/branch/master/graph/badge.svg)](https://codecov.io/gh/tryolabs/luminoth)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

Luminoth is an open source toolkit for **computer vision**. Currently, we support object detection and image classification, but we are aiming for much more. It is built in Python, using [TensorFlow](https://www.tensorflow.org/) and [Sonnet](https://github.com/deepmind/sonnet).

![Example of Object Detection](https://luminoth.ai/images/screen.png?v=1)

> **DISCLAIMER**: Luminoth is still alpha-quality release, which means the internal and external interfaces (such as command line) are very likely to change as the codebase matures.

# Installation
Luminoth currently supports Python 2.7 and 3.4–3.6.

## Pre-requisites
If you want **GPU support**, you should install the GPU version of [TensorFlow](https://www.tensorflow.org/install/).
If TensorFlow is is already installed, Luminoth will use that version (no matter if CPU or GPU versions).

## Installing Luminoth
Just install from PyPI:

```bash
$ pip install luminoth
```

## Installing from source

First, clone the repo on your machine and then install with `pip`:

```
$ git clone https://github.com/tryolabs/luminoth.git
$ cd luminoth
$ pip install -e .
```

## Check that the installation worked

Simply run `lumi --help`.

# Supported models

Currently, we support the following models:

* **Object Detection**
    * [Faster R-CNN](https://arxiv.org/abs/1506.01497)

We are planning on adding support for more models in the near future, such as [SSD](https://arxiv.org/abs/1512.02325), [YOLO](https://arxiv.org/abs/1506.02640) and [Mask R-CNN](https://arxiv.org/abs/1703.06870).

Moreover, we are also working on providing **pre-trained checkpoints** on popular datasets such as [Pascal VOC2012](http://host.robots.ox.ac.uk:8080/pascal/VOC/voc2012/index.html).

# Usage

There is one main command line interface which you can use with the `lumi` command. Whenever you are confused on how you are supposed to do something just type:

`lumi --help` or `lumi <subcommand> --help`

and a list of available options with descriptions will show up.

## Working with datasets
See [DATASETS](./docs/DATASETS.md).

## Training

Check our [TRAINING](./docs/TRAINING.md) on how to train locally or in Google Cloud.

## Visualizing results

We strive to get useful and understandable summary and graph visualizations. We consider them to be essential not only for monitoring (duh!), but for getting a broader understanding of what's going under the hood. The same way it is important for code to be understandable and easy to follow, the computation graph should be as well.

By default summary and graph logs are saved to `jobs/` under the current directory. You can use TensorBoard by running:

```
tensorboard --logdir path/to/jobs
```

## Why the name?
> The Dark Visor is a Visor upgrade in Metroid Prime 2: Echoes. Designed by the **Luminoth** during the war, it was used by the Champion of Aether, A-Kul, to penetrate Dark Aether's haze in battle against the Ing.
>
> -- [Dark Visor - Wikitroid](http://metroid.wikia.com/wiki/Dark_Visor)
>

# License
Copyright © 2018, [Tryolabs](https://tryolabs.com).
Released under the [BSD 3-Clause](LICENSE).
