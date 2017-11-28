import tensorflow as tf

from luminoth.models.retina.subnet import Subnet


class ClassSubnet(Subnet):
    """Classification subnet for Retina.

    One of these (or the same one, depending on whether or not we share
    weights) will hang from each FPN level.
    """
    def __init__(self, config, num_anchors, num_classes, name='class_subnet'):
        num_final_chns = num_anchors * (num_classes)
        # This is the pi parameter in the paper.
        init_bias_foreground = config.final.init_bias.foreground
        final_bias = tf.constant_initializer(init_bias_foreground)
        super(ClassSubnet, self).__init__(
            config, num_final_chns, final_bias=final_bias, name=name
        )
        self._config = config
