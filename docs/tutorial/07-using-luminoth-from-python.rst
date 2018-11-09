.. _tutorial/07-using-luminoth-from-python:

Using Luminoth from Python
==========================

Calling Luminoth from your Python app is very straightforward. You can even make use of
helper functions to visualize the bounding boxes.

.. code-block:: python

   from luminoth import Detector, read_image, vis_objects

   image = read_image('traffic-image.png')

   # If no checkpoint specified, will assume `accurate` by default. In this case,
   # we want to use our traffic checkpoint. The Detector can also take a config
   # object.
   detector = Detector(checkpoint='traffic')

   # Returns a dictionary with the detections.
   objects = detector.predict(image)

   print(objects)

   vis_objects(image, objects).save('traffic-out.png')

----

This was the end of the tutorial! Hope you enjoyed :)
