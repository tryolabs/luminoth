import tensorflow as tf

from luminoth.models.retina.subnet import Subnet


class ClassSubnet(Subnet):
    """Classification subnet for Retina.

    One of these (or the same one, depending on whether or not we share
    weights) will hang from each FPN level.
    """
    def __init__(self, config, num_anchors, num_classes, name='class_subnet'):
        num_final_chns = num_anchors * (num_classes + 1)
        # Pi as the parameter in the paper.
        init_bias_foreground = config.final.init_bias.foreground
        # This isn't mentioned in the paper because they deal with the special
        # case of binary classification where it isn't necessary.
        init_bias_background = config.final.init_bias.background
        background_bias = init_bias_background
        foreground_bias = init_bias_foreground
        final_bias = tf.constant_initializer(
            ([background_bias] + ([foreground_bias] * (num_classes)))
            * num_anchors
        )
        super(ClassSubnet, self).__init__(
            config, num_final_chns, final_bias=final_bias, name=name
        )
        self._config = config
