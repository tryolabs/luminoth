Detector
========

Download [Pascal VOC2007](http://host.robots.ox.ac.uk:8080/pascal/VOC/voc2007/)'s data, both [train/val](http://host.robots.ox.ac.uk:8080/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar) and [test](http://host.robots.ox.ac.uk:8080/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar) datasets, and extract them; they will merge into single folder named `VOCdevkit`. Then copy the contents of `VOCdevkit/VOC2007` into `datasets/voc`.


Prepare the dataset by running the following line. This will transform the data into three `.tfrecords` files (train, validation, and training) and place them in `datasets/voc/tf/`.

```
$ python -m detector voc
```

Run the training with:

```
$ python -m detector train
```

Run the evaluator (separate process) with:

```
$ python -m detector evaluate --split=val
```


FRCNN
=====

To build Cython extensions:
```
$ python setup.py build_ext --inplace
```
