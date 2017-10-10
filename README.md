Luminoth
========

[![Build Status](https://travis-ci.org/tryolabs/luminoth.svg?branch=master)](https://travis-ci.org/tryolabs/luminoth)
[![codecov](https://codecov.io/gh/tryolabs/luminoth/branch/master/graph/badge.svg)](https://codecov.io/gh/tryolabs/luminoth)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

> The Dark Visor is a Visor upgrade in Metroid Prime 2: Echoes. Designed by the **Luminoth** during the war, it was used by the Champion of Aether, A-Kul, to penetrate Dark Aether's haze in battle against the Ing.
>
> -- [Dark Visor - Wikitroid](http://metroid.wikia.com/wiki/Dark_Visor)

# What is Luminoth?

Luminoth is a computer vision toolkit made with [Tensorflow](https://www.tensorflow.org/) and [Sonnet](https://deepmind.github.io/sonnet/). Our main objective is to create tools and code to easily train and use deep learning models for computer vision problems.

- Code that is both easy to understand and easy to extend.
- Out-of-the-box state of the art models.
- Straightforward implementations with TensorBoard support.
- Cloud integration for training and deploying.

> **DISCLAIMER**: This is currently a pre-pre-alpha release, we decided to open-source it up for those inquisive minds that don't mind getting their hands dirty with rough edges of code.

## Why Luminoth

We started building Luminoth at [Tryolabs](https://tryolabs.com/) after realizing we always ended up rewriting many of the common Tensorflow boilerplate code and models over and over. Instead of just building a cookie-cutter for Tensorflow we started to think about what other features we could benefit from, and how an ideal toolkit would look like.

## Why Tensorflow (and why Sonnet)?

It is indisputable that TensorFlow is currently the most mature Deep Learning framework, and even though we love (truly love) other frameworks as well, especially [PyTorch](http://pytorch.org), our customers demand stable and production ready Machine Learning solutions.

[Sonnet](https://deepmind.github.io/sonnet/) fits perfectly with our mission to build code that is easy to follow and to extend. It is tricky to build a computation graph that is abstract and low-level at the same time to allows us to build complex models, and luckily Sonnet is a library that provides just that.

# Installation
Luminoth currently supports Python 2.7 and 3.4–3.6.

If [TensorFlow](https://www.tensorflow.org) and [Sonnet](https://github.com/deepmind/sonnet) are already installed, Luminoth will use those versions.

## Install with CPU support
Just run:
```bash
$ pip install luminoth
```

This will install the CPU versions of TensorFlow & Sonnet if you don't have them.

## Install with GPU support

1. [Install TensorFlow](https://www.tensorflow.org/install/) with GPU support.
2. [Install Sonnet](https://github.com/deepmind/sonnet#installation) with GPU support:
    ```bash
    $ pip install dm-sonnet-gpu
    ```
3. Install Luminoth from PyPI:
    ```bash
    $ pip install luminoth
    ```

## Install from source

First, clone the repo on your machine and then install with `pip`:

```
$ git clone https://github.com/tryolabs/luminoth.git
$ cd luminoth
$ pip install -e .
```

## Check that the installation worked

Simply run `lumi --help`.

# Supported models

Currently we are focusing on object detection problems, and have a fully functional version of [Faster RCNN](https://arxiv.org/abs/1506.01497). There are more models in progress (SSD and Mask RCNN to name a couple), and we look forward to opening up those implementations.

# Usage

There is one main command line interface which you can use with the `lumi` command. Whenever you are confused on how you are supposed to do something just type:

`lumi --help` or `lumi <subcommand> --help`

and a list of available options with descriptions will show up.

## Using existing dataset
See [DATASETS.md](./docs/DATASETS.md).

## Training

Check our [TRAINING.md](./docs/TRAINING.md) on how to train locally or in Google Cloud.

## Visualizing results

We strive to get useful and understandable summary and graph visualizations. We consider them to be essential not only for monitoring (duh!), but for getting a broader understanding of whats going under the hood. The same way it is important for code to be understandable and easy to follow, the computation graph should be as well.

By default summary and graph logs are saved to `/tmp/luminoth`. You can use TensorBoard by running:

```
tensorboard --logdir /tmp/luminoth
```
