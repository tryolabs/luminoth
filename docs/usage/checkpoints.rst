.. _usage/checkpoints:

Working with checkpoints
========================

TODO: Explain the rationale behind checkpoints, and expand each section.

List the checkpoints available on the system::

  $ lumi checkpoint list
  ================================================================================
  |           id |                  name |       alias | source |         status |
  ================================================================================
  | 48ed2350f5b2 |   Faster R-CNN w/COCO |    accurate | remote | NOT_DOWNLOADED |
  | e3256ffb7e29 |      SSD w/Pascal VOC |        fast | remote | NOT_DOWNLOADED |
  ================================================================================

Inspect a checkpoint::

  $ lumi checkpoint info accurate
  Faster R-CNN w/COCO (48ed2350f5b2, accurate)
  Base Faster R-CNN model trained with the full COCO dataset.

  Model used: fasterrcnn
  Dataset information
      Name: COCO
      Number of classes: 80

  Creation date: 2018-03-21T20:04:59.785711
  Luminoth version: v0.1.0

  Source: remote (NOT_DOWNLOADED)
  URL: https://github.com/tryolabs/luminoth/releases/download/v0.0.3/48ed2350f5b2.tar

Refresh the remote checkpoint index::

  $ lumi checkpoint refresh
  Retrieving remote index... done.
  2 new remote checkpoints added.

Download a remote checkpoint::

  $ lumi checkpoint download accurate
  Downloading checkpoint...  [####################################]  100%
  Importing checkpoint... done.
  Checkpoint imported successfully.

Create a checkpoint::

  $ lumi checkpoint create config.yml -e name='Faster R-CNN with cars' -e alias=cars
  Creating checkpoint for given configuration...
  Checkpoint b5c140450f48 created successfully.

Edit a checkpoint::

  $ lumi checkpoint edit b5c140450f48 -e 'description=Model trained with COCO cars.'

Delete a checkpoint::

  $ lumi checkpoint delete b5c140450f48
  Checkpoint b5c140450f48 deleted successfully.

Export a checkpoint into a tar file, for easy sharing::

  $ lumi checkpoint export 48ed2350f5b2
  Checkpoint 48ed2350f5b2 exported successfully.

Import a previously-exported checkpoint::

  $ lumi checkpoint import 48ed2350f5b2.tar
