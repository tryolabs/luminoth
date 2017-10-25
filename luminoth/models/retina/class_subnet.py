import numpy as np

from luminoth.models.retina.subnet import Subnet


class ClassSubnet(Subnet):
    def __init__(self, config, num_anchors, num_classes, name='box_subnet'):
        num_final_chns = num_anchors * (num_classes + 1)
        pi = config.final.pi
        final_bias = - np.log((1 - pi) / pi)
        super(ClassSubnet, self).__init__(
            config, num_final_chns, final_bias=final_bias, name=name)
        self._config = config
