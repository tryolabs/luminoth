import sonnet as snt
import tensorflow as tf
import numpy as np

class RCNN(snt.AbstractModule):
    """RCNN """
    def __init__(self, num_classes, layer_sizes=[4096, 4096], name='rcnn'):
        super(RCNN, self).__init__(name=name)
        self._num_classes = num_classes
        self._layer_sizes = layer_sizes
        self._activation = tf.nn.relu
        self._dropout_keep_prob = 0.6
        self._instantiate_layers()

    def _instantiate_layers(self):
        with self._enter_variable_scope():

            # Define initializaers/partitioners/regualizers
            self._layers = [
                snt.Linear(
                    layer_size,
                    name="fc_{}".format(i),
                    initializers=self._initializers,
                    partitioners=self._partitioners,
                    regularizers=self._regularizers,
                    use_bias=self.use_bias
                )
                for i, layer_size in enumerate(self._layer_sizes)
            ]

            self._classifier_layer = snt.Linear(
                self._num_classes, name="fc_classifier"
            )

            # TODO: Not random initializer
            self._bbox_layer = snt.Linear(
                self._num_classes * 4, name="fc_bbox"
            )


    def _build(self, pooled_layer):
        """
        TODO: El pooled layer es el volumen con todos los ROI o es uno por cada ROI?
        """
        net = pooled_layer
        for i, layer in enumerate(self._layers):
            net = layer(net)
            net = self._activation(net)
            net = tf.nn.dropout(net, keep_prob=self._dropout_keep_prob)

        classification_net = self._classifier_layer(net)
        classification_prob = tf.nn.softmax(classification_net)
        bbox_net = self._bbox_layer(net)

        return classification_prob, bbox_net




