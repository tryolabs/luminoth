import tensorflow as tf
import sonnet as snt

from luminoth.models.base import BaseNetwork


class FPN(BaseNetwork):
    def __init__(self, config, parent_name=None, name='fpn_builder'):
        super(FPN, self).__init__(config, name=name)
        self._config = config
        self._parent_name = parent_name
        # Copy the endpoints to prevent unwanted behaviour.
        self._endpoints = config.endpoints[:]
        self._num_channels = config.num_channels
        for i, endpoint in enumerate(self._endpoints):
            self._endpoints[i] = '{}/{}/{}'.format(
                self.module_name, config.architecture, self._endpoints[i]
            )
            if parent_name:
                self._endpoints[i] = '{}/{}'.format(
                    parent_name, self._endpoints[i]
                )

    def _build(self, inputs, is_training=True):
        base_pred = super(FPN, self)._build(
            inputs, is_training=is_training
        )
        end_points_unordered = self._get_endpoint_list(base_pred)
        # TODO: sort end_points.
        end_points = end_points_unordered

        # Create one 1x1 Conv layer for each pyramid level.
        conversors = []
        for i in range(len(end_points)):
            conversors.append(snt.Conv2D(
                output_channels=self._num_channels,
                kernel_shape=[1, 1], name='fpn_level_{}'.format(i)
            ))

        try:
            fpn_levels = [conversors[0](end_points[0])]
        except IndexError:
            raise ValueError('No valid endpoints to build FPN.')

        for i, end_point in enumerate(end_points):
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
        end_points_dict = base_pred['end_points']
        end_points = []
        for key in self._endpoints:
            try:
                end_points.append(end_points_dict[key])
            except KeyError:
                tf.logging.warn('Illegal endpoint name "{}"'.format(key))
        if len(end_points) < 1:
            raise ValueError('No legal endpoint names for FPN.')
        return end_points
