import tensorflow as tf
import sonnet as snt

from luminoth.models.base import BaseNetwork


class FPN(BaseNetwork):
    """Module for building a Feature Pyramid Networks.

    As seen in "Feature Pyramid Networks for Object Detection" (2017).
    """
    def __init__(self, config, parent_name=None, name='fpn_builder'):
        super(FPN, self).__init__(config, name=name)
        self._config = config
        self._parent_name = parent_name
        # Copy the endpoints to prevent unwanted behaviour.
        self._endpoints = config.endpoints[:]
        self._num_channels = config.num_channels

    def _build(self, inputs, is_training=True):
        """This takes an image and returns a list with FPN levels.

        Args:
            inputs: (1, height, width, 3)
            is_training: Bool.

        Returns:
            fpn_levels: list of (1, H, W, 3) tensors.
        """
        inputs.set_shape([1, None, None, 3])
        base_pred = super(FPN, self)._build(
            inputs, is_training=is_training
        )
        end_points_unordered = self._get_endpoint_list(base_pred)
        # TODO: sort end_points.
        # We make them public because we may need them for the layers that use
        # FPN (e.g. Retina)
        self.end_points = end_points_unordered

        # Create one 1x1 Conv layer for each pyramid level.
        conversors = []
        for i in range(len(self.end_points)):
            conversors.append(snt.Conv2D(
                output_channels=self._num_channels,
                kernel_shape=[1, 1], name='fpn_level_{}'.format(i)
            ))

        try:
            fpn_levels = [conversors[0](self.end_points[0])]
        except IndexError:
            raise ValueError('No valid endpoints to build FPN.')

        for i, end_point in enumerate(self.end_points):
            if i == 0:
                continue
            previous_level = fpn_levels[i - 1]
            upsampled = tf.image.resize_nearest_neighbor(
                previous_level, tf.shape(end_point)[1:3],
                name='upsample_endpoint_{}'.format(i)
            )
            converted_endpoint = conversors[i](end_point)

            new_level = tf.add(upsampled, converted_endpoint)
            fpn_levels.append(new_level)

        return fpn_levels

    def _get_endpoint_list(self, base_pred):
        """
        Args:
            base_pred: dict. Output of the base network.

        Returns:
            end_points: List of endpoints of the base that we will consider.
        """
        end_points_dict = base_pred['end_points']
        end_points = []
        for name in self._endpoints:
            for key, value in end_points_dict.items():
                if key.endswith(name):
                    end_points.append(value)
                    continue
        if len(end_points) < 1:
            raise ValueError('No legal endpoint names for FPN.')
        return end_points

    def get_trainable_vars(self, train_base):
        trainable_vars = snt.get_variables_in_module(self)
        if train_base:
            trainable_vars += super(FPN, self).get_trainable_vars()
        return trainable_vars
