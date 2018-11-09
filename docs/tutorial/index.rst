.. _tutorial/index:

Tutorial: real world object detection with Luminoth
===================================================

In this tutorial, we will learn the workings of *Luminoth* by using it in practice to
solve a real world object detection problem.

As our case study, we will be building a model able to recognize cars, pedestrians, and
other objects which a self-driving car would need to detect in order to properly function.
We will have our model ready for that and see it how to apply it to images and video. We
will not, however, add any tracking capabilities.

To follow along easier and not invest many hours each time we want to run the training
process, we will build a small toy dataset and show how things go from there, giving tips
on the things you need to look at when training a model with a larger dataset.

First, check the :ref:`usage/installation` section and make sure you have a working
install.

.. toctree::
   :maxdepth: 2

   01-first-steps
   02-building-custom-traffic-dataset
   03-training-the-model
   04-visualizing-the-training-process
   05-evaluating-models
   06-creating-own-checkpoints
   07-using-luminoth-from-python
