Detector
========

Download Pascal VOC2007 data [here](http://host.robots.ox.ac.uk:8080/pascal/VOC/voc2007/), both train/val and test, and merge into single folder.

Prepare the dataset with, this will transform the data into three `.tfrecords` files.

```
$ python -m detector.voc --data-dir=/path/to/datasets/voc --output-dir=/path/to/output
```

Run the training with:

```
$ python -m detector --data-dir=/path/to/tfrecordsdir
```