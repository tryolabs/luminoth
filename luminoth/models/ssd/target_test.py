import numpy as np
import tensorflow as tf

from easydict import EasyDict
from luminoth.models.ssd.target import SSDTarget
from luminoth.utils.bbox_transform import encode


class TargetTest(tf.test.TestCase):

    def setUp(self):
        super(TargetTest, self).setUp()
        self._config = EasyDict({
            'hard_negative_ratio': 2.,
            'foreground_threshold': 0.5,
            'background_threshold_high': 0.2,
            'background_threshold_low': 0.0,
            'variances': [0.1, 0.2]
        })
        self._equality_delta = 1e-03
        self._shared_model = SSDTarget(
            self._config, self._config.variances, seed=0
        )
        self.anchors = np.array([
            [40.341, 40.3, 80.0, 80.],
            [0., 0., 15., 15.],
            [45., 45., 125., 220.],
            [110., 190., 240., 270.],
            [130., 210., 240., 270.],
            [1., 1., 299., 299.],
            [220., 50., 280., 200.],
            [200., 210., 210., 220.],
            [10., 20., 10., 25.],
            [20., 10., 25., 10.]
        ])

        tf.reset_default_graph()

    def _run_ssd_target(self, model, probs, all_anchors, gt_boxes):
        """Runs an instance of SSDTarget

        Args:
            model: an SSDTarget model.
            gt_boxes: a Tensor holding the ground truth boxes, with shape
                (num_gt, 5). The last value is the class label.
            proposed_boxes: a Tensor holding the proposed boxes. Its shape is
                (num_proposals, 4). The first value is the batch number.

        Returns:
            The tuple returned by SSDTarget._build().
        """
        ssd_target_net = model(probs, all_anchors, gt_boxes)
        with self.test_session() as sess:
            return sess.run(ssd_target_net)

    def _create_a_2_gt_box_sample(self):
        """Creates a testing sample

        Creates sample with 2 gt_boxes, with 3 true predictions and 7 false.
        Contains 3 classes.
        gt_box format = [xmin, ymin, xmax, ymax, label]
        anchors format = [xmin, ymin, xmax, ymax]
        """
        gt_boxes = np.array(
            [[10, 10, 50, 50, 0], [120, 200, 250, 280, 2]], dtype=np.float32
        )

        probs = np.array([
            [.1, .2, .7],
            [.8, .1, .1],
            [.3, .3, .4],
            [.1, .5, .4],
            [.8, .1, .1],
            [.3, .3, .4],
            [.1, .5, .4],
            [.8, .1, .1],
            [.3, .3, .4],
            [.1, .5, .4]
        ], dtype=np.float32)

        target_offsets = np.zeros_like(self.anchors)
        target_offsets[0, :] = encode(
            np.expand_dims(self.anchors[0], axis=0),
            np.expand_dims(gt_boxes[0], axis=0),
            self._config.variances
        )
        target_offsets[3, :] = encode(
            np.expand_dims(self.anchors[3], axis=0),
            np.expand_dims(gt_boxes[1], axis=0),
            self._config.variances
        )
        target_offsets[4, :] = encode(
            np.expand_dims(self.anchors[4], axis=0),
            np.expand_dims(gt_boxes[1], axis=0),
            self._config.variances
        )

        target_classes = np.array([1., 0., 0., 3., 3., 0., 0., -1., 0., 0.],
                                  dtype=np.float32)
        return probs, gt_boxes, target_offsets, target_classes

    def _create_a_1_gt_box_sample(self):
        """Creates a testing sample

        Creates sample with 1 gt_box, with 2 true predictions and 8 false.
        Contains 3 classes.
        gt_box format = [xmin, ymin, xmax, ymax, label]
        anchors format = [xmin, ymin, xmax, ymax]
        """
        gt_boxes = np.array([[20, 20, 80, 80, 0]], dtype=np.float32)
        probs = np.array([
            [.1, .2, .7],
            [.8, .1, .1],
            [.3, .3, .4],
            [.1, .5, .4],
            [.8, .1, .1],
            [.3, .3, .4],
            [.1, .5, .4],
            [.8, .1, .1],
            [.3, .3, .4],
            [.1, .5, .4]
        ], dtype=np.float32)

        target_offsets = np.zeros_like(self.anchors)
        target_offsets[0, :] = encode(
            np.expand_dims(self.anchors[0], axis=0),
            np.expand_dims(gt_boxes[0], axis=0),
            self._config.variances
        )

        target_classes = np.array(
            [1., -1., -1., 0., -1., -1., 0., -1., -1., -1.], dtype=np.float32
        )
        return probs, gt_boxes, target_offsets, target_classes

    def testHardCase(self):
        """Tests a hard case with batch_size == 6"""

        probs, gt_boxes, target_offsets_test, target_classes_test \
            = self._create_a_2_gt_box_sample()

        # Run test through target
        target_classes, target_offsets = self._run_ssd_target(
            self._shared_model, probs, self.anchors, gt_boxes
        )

        np.testing.assert_allclose(
            target_offsets_test, target_offsets, rtol=self._equality_delta)
        np.testing.assert_allclose(
            target_classes, target_classes_test, rtol=self._equality_delta)


if __name__ == '__main__':
    tf.test.main()
