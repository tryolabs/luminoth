# Datasets

If a pre-trained model for the task you want to perform is not available, you can train Luminoth with an existing open dataset, or your own.

The first step in training Luminoth is converting your dataset to TensorFlow's `.tfrecords` format. This ensures that no matter what image or annotation formats the original dataset uses, it will be transformed to something that Luminoth can understand and process efficiently, either while training locally or in the cloud.

For this purpose, Luminoth provides the `lumi dataset transform` command, which includes support for some of the most well-known datasets for object detection and classification tasks.

## Supported datasets

- [Pascal VOC2012](http://host.robots.ox.ac.uk:8080/pascal/VOC/voc2012/index.html)

```
$ lumi dataset transform --type pascal --data-dir ~/dataset/pascal/ --output-dir ~/dataset/pascal/tf/ --split train --split val
```

- [ImageNet](http://image-net.org/download)

```
$ lumi dataset transform --type imagenet --data-dir ~/dataset/imagenet/ --output-dir ~/dataset/imagenet/tf/ --split train --split val
```

- [COCO](http://cocodataset.org/#download)

```
$ lumi dataset transform --type coco --data-dir ~/dataset/coco/ --output-dir ~/dataset/coco/tf --split train --split val
```

## Limiting the dataset
During development, it is often useful to verify that the model can actually overfit a small dataset.

You can use the `--limit-examples` and `--limit-classes` options for this.

For more information, try `lumi dataset transform --help`.

## Supporting your own dataset
TODO guidelines on how to write your own conversion tool
