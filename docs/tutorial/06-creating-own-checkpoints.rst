.. _tutorial/06-creating-own-checkpoints:

Creating and sharing your own checkpoints
=========================================

After the model is trained to your satisfaction, it is very useful to actually create a
**checkpoint** that makes it straightforward to use your model.

Creating a checkpoint
---------------------

We can create checkpoints and set some metadata like name, alias, etc. This time, we are
going to create the checkpoint for our traffic model:

.. code-block:: bash

   lumi checkpoint create \
       config.yml \
       -e name="OpenImages Traffic" \
       -e alias=traffic

After running this, you should get an output similar to this:

.. code-block:: text

   Creating checkpoint for given configuration...
   Checkpoint cb0e5d92a854 created successfully.


You can verify that you do indeed have the checkpoint when running ``lumi checkpoint
list``, which should get you an output similar to this:

.. code-block:: text

   ================================================================================
   |           id |                  name |       alias | source |         status |
   ================================================================================
   | e1c2565b51e9 |   Faster R-CNN w/COCO |    accurate | remote |     DOWNLOADED |
   | aad6912e94d9 |      SSD w/Pascal VOC |        fast | remote |     DOWNLOADED |
   | cb0e5d92a854 |    OpenImages Traffic |     traffic |  local |          LOCAL |
   ================================================================================


Moreover, if you inspect the ``~/.luminoth/checkpoints/`` folder, you will see that now you
have a folder that corresponds to your newly created checkpoint. Inside this folder are
the actual weights of the model, plus some metadata and the configuration file that was
used during training.

Sharing checkpoints
-------------------

Exporting a checkpoint as a single file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Simply run ``lumi checkpoint export cb0e5d92a854``. You will get a file named
``cb0e5d92a854.tar`` in your current directory, which you can easily share to somebody else.

Importing a checkpoint file
^^^^^^^^^^^^^^^^^^^^^^^^^^^

By running ``lumi checkpoint import cb0e5d92a854.tar``, the checkpoint will be listed
locally. Note that this will fail if the checkpoint already exists, as expected (you can
use ``lumi checkpoint delete`` if you want to try this anyway).

You can now use it very easily, for example we can reference our checkpoint using its
alias by running ``lumi server web --checkpoint traffic``. Neat!

----

Next: :ref:`tutorial/07-using-luminoth-from-python`
