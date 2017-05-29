import sonnet as snt


from .rpn import RPN
from .anchor import AnchorTarget

class FasterRCNN(snt.AbstractModule):
    """Faster RCNN Network"""
    def __init__(self, config, name='fasterrcnn'):
        super(FasterRCNN, self).__init__(name=name)

        self._cfg = config

        self._pretrained = PreTrained(self._cfg.PRETRAINED_ARCHITECTURE)
        self._rpn = RPN(self._cfg.ANCHOR_SCALES, self._cfg.ANCHOR_RATIOS)
        self._anchor = AnchorTarget(self._cfg.ANCHOR_SCALES, self._cfg.ANCHOR_RATIOS)

        # TODO: Seguir con "layers"/secciones.

    def _build(self, input):
        # TODO: Algo asi?
        pretrained_output = self._pretrained(input)
        rpn_cls_prob, rpn_bbox_pred = self._rpn(pretrained_output)
        self._anchor(rpn_cls_prob)
