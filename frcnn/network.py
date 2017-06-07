import sonnet as snt


from .rpn import RPN
from .anchor import AnchorTarget
from .proposal import Proposal

class FasterRCNN(snt.AbstractModule):
    """Faster RCNN Network"""
    def __init__(self, config, num_classes=None, name='fasterrcnn'):
        super(FasterRCNN, self).__init__(name=name)

        self._cfg = config
        self._is_training = is_training
        self._num_classes = num_classes

        with self._enter_variable_scope():
            self._pretrained = VGG()
            self._rpn = RPN(
                self._cfg.ANCHOR_SCALES, self._cfg.ANCHOR_RATIOS,
                is_training=self._is_training
            )
            self._anchor_target = AnchorTarget(self._rpn.anchors)
            self._proposal = Proposal(self._rpn.anchors)
            self._roi_pool = ROIPoolingLayer()
            self._rcnn = RCNN(self._num_classes)

    def _build(self, image, gt_boxes, is_training=True):
        pretrained_output = self._pretrained(image)
        rpn_layers = self._rpn(pretrained_output, is_training=is_training)
        labels, bbox_targets = self._anchor_target(rpn_layers['rpn_cls_score_reshape'], gt_boxes)
        blob, scores = self._proposal(rpn_layers['rpn_cls_prob'], rpn_layers['rpn_bbox_pred'])
        roi_pool = self._roi_pool(blob, pretrained_output)
        classification_prob, bbox_net = self._rcnn(roi_pool)

        return classification_prob, bbox_net
