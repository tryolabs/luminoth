import tensorflow as tf
import sonnet as snt

from luminoth.utils.vars import get_activation_function


class Subnet(snt.AbstractModule):
    """Simple convolutional subnets for use with Retina.

    Simply a series of Conv2D layers with activations in the middle.
    """
    def __init__(self, config, num_final_channels, final_bias, prefix=None,
                 name='subnet'):
        super(Subnet, self).__init__(name=name)
        if prefix is None:
            prefix = name
        self._config = config
        self._num_final_channels = num_final_channels

        self._prefix = prefix

        self._final_bias = final_bias

        self._hidden_activation = get_activation_function(
            config.hidden.activation
        )
        self._final_activation = get_activation_function(
            config.final.activation
        )

    def _build(self, fpn_level):
        """
        Args:
            fpn_level: (1, H, W, 3)

        Returns:
            A score bank with rank 4. The meaning depends on whether this will
            be a class subnet or a box subnet.
        """
        layers = []
        for i in range(self._config.hidden.depth):
            # TODO: consider making the initializer configurable, although it
            # may not be the best idea, as most initializations don't converge.
            new_layer = snt.Conv2D(
                output_channels=self._config.hidden.channels,
                kernel_shape=self._config.hidden.kernel_shape,
                initializers={
                    'w': tf.random_normal_initializer(
                        mean=0.0, stddev=0.01
                    ),
                },
                name='{}_hidden_{}'.format(self._prefix, i)
            )
            layers.append(new_layer)

        pred = fpn_level
        for layer in layers:
            pred = self._hidden_activation(layer(pred))

        final_layer = snt.Conv2D(
            output_channels=self._num_final_channels,
            kernel_shape=self._config.final.kernel_shape,
            initializers={
                'w': tf.random_normal_initializer(
                    mean=0.0, stddev=0.01
                ),
                'b': self._final_bias
            },
            name='{}_final'.format(self._prefix)
        )
        return self._final_activation(final_layer(pred))
