Luminoth
========

> The Dark Visor is a Visor upgrade in Metroid Prime 2: Echoes. Designed by the **Luminoth** during the war, it was used by the Champion of Aether, A-Kul, to penetrate Dark Aether's haze in battle against the Ing.
>
> -- [Dark Visor - Wikitroid](http://metroid.wikia.com/wiki/Dark_Visor)

# What is Luminoth?

Luminoth is a computer vision toolkit made with [Tensorflow](https://www.tensorflow.org/) and [Sonnet](https://deepmind.github.io/sonnet/). Our main objective is to create tools and code to easily train and use deep learning models for computer vision problems.

- Code that is both, easy to understand and easy to extend.
- Out-of-the-box state of the art models.
- Straightforward implementations with TensorBoard support.
- Cloud integration for training and deploying.

> **DISCLAIMER**: This is currently a pre-pre-alpha release, we decided to open-source it up for those inquisive minds that don't mind getting their hands dirty with rough edges of code.

## Why Luminoth

We started building Luminoth at [Tryolabs](https://tryolabs.com/) after realizing we always ended up rewriting many of the common Tensorflow boilerplate code and models over and over. Instead of just building a cookie-cutter for Tensorflow we started to think about what other features we could benefit from, and how would an ideal toolkit would look like.

## Why Tensorflow (and why Sonnet)?

It is indisputable that TensorFlow is currently the most mature Deep Learning framework, even though we love (truly love) other frameworks as well, especially [PyTorch](http://pytorch.org), our customers demand stable and production ready Machine Learning solutions.

[Sonnet](https://deepmind.github.io/sonnet/) fits perfectly with our mission to build code that is easy to follow and to extend. It is tricky to build a computation graph that is abstract and low-level at the same time to allows us to build complex models, and luckily Sonnet is a library that provides just that.

# Installation

Luminoth currently supports for both Python3 and Python2.7.

To install clone the repo on your machine:

```
git clone https://github.com/tryolabs/luminoth.git; cd luminoth;
```

and then run

```
python setup.py install
```

This should install all required dependencies.

You can check if it's working by running

```
lumi --help
```

# Supported models

Currently we are focusing on object detection problems, and have a fully functional version of [Faster RCNN](https://arxiv.org/abs/1506.01497). There are more models in progress (SSD and Mask RCNN to name a couple), and we look forward to opening up those implementations.

# Usage

There is one main command line interface which you can use with the `lumi` command. Whenever you are confused on how you are supposed to do something just type:

`lumi --help` or `lumi <subcommand> --help`

and a list of available options with descriptions will show up.

## Datasets

Convert datasets to TensorFlow's *`.tfrecords`* for efficient processing using the computation graphs (and for cloud support).

- [Pascal VOC2012](http://host.robots.ox.ac.uk:8080/pascal/VOC/voc2012/index.html)

```
lumi dataset voc --data-dir ~/dataset/voc/ --output-dir ~/dataset/voc/tf/
```

- [ImageNet](http://image-net.org/download)

```
lumi dataset imagenet --data-dir ~/dataset/imagenet/ --output-dir ~/dataset/imagenet/tf/
```

- [COCO](http://mscoco.org/dataset/#download)

```
lumi dataset coco --data-dir ~/dataset/coco/ --output-dir ~/dataset/coco/tf/
```

## Training

Check our [TRAINING.md](./TRAINING.md) on how to train locally or in Google Cloud.

## Visualizing results

We strive to get useful and understandable summary and graph visualizations. We consider them to be essential not only for monitoring (duh!), but for getting a broader understanding of whats going under the hood. The same way it is important for code to be understandable and easy to follow, the computation graph should be as well.

By default summary and graph logs are saved to `/tmp/luminoth`. You can use TensorBoard by running:

```
tensorboard --logdir /tmp/luminoth
```
