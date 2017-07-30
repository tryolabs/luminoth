Luminoth
========

> The Dark Visor is a Visor upgrade in Metroid Prime 2: Echoes. Designed by the **Luminoth** during the war, it was used by the Champion of Aether, A-Kul, to penetrate Dark Aether's haze in battle against the Ing.
>
> -- [Dark Visor - Wikitroid](http://metroid.wikia.com/wiki/Dark_Visor)


# What is Luminoth?

Luminoth is a computer vision toolkit made with Tensorflow and Sonnet. Our objective is to create tools to train and use deep learning models while at the same time writing code that is both easy to read (even for a Machine Learning novice) and easy to mantain and extend.


# Installation

We currently only support Python 3 (tested on 3.6). Before installing Luminoth you need to manually install [Sonnet](https://github.com/deepmind/sonnet) which does not currently have binary builds for easy installation with `pip`.

After installing Sonnet you can proceed by running:

```
$ python setup.py install
```

This should install all required dependencies.

# Usage

There one main command line interface which you can use with the `lumi` command.

## Datasets management

Convert the [Pascal VOC2007](http://host.robots.ox.ac.uk:8080/pascal/VOC/voc2007/) data to Tensorflow's *tfrecords*.

```
lumi dataset voc --data-dir ~/dataset/voc/ --output-dir ~/dataset/voc/tf/
```

## Train

Copy FasterRCNN's base_config.yml and modify parameters.

```
lumi train fasterrcnn --config fasterrcnn-custom.yml
```

## Evaluate


Download [Pascal VOC2007](http://host.robots.ox.ac.uk:8080/pascal/VOC/voc2007/)'s data, both [train/val](http://host.robots.ox.ac.uk:8080/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar) and [test](http://host.robots.ox.ac.uk:8080/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar) datasets, and extract them; they will merge into single folder named `vocdevkit`. then copy the contents of `vocdevkit/voc2007` into `datasets/voc`.
