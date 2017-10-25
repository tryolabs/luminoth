import tensorflow as tf
import numpy as np

from easydict import EasyDict

from luminoth.models.retina import Retina
from luminoth.models.retina.retina_target import RetinaTarget
from luminoth.utils.config import get_base_config


class RetinaTargetTest(tf.test.TestCase):
    def setUp(self):
        super(RetinaTargetTest, self).setUp()
        self._config = get_base_config(Retina)['model']['target']
        self._num_classes = 4
        self._placeholder_label = 2
        self._model = RetinaTarget(self._config, self._num_classes)

        self._equality_delta = 1.e-3

    def _run_retina_target(self, model, gt_boxes, anchors):
        """Run an instance of RetinaTarget

        Args:
            model: RetinaTarget model.
            gt_boxes: Tensor holding the ground truth boxes, with shape
                (num_gt, 5). The last value is the class label.
            anchors: Tensor holding the anchor. Its shape is
                (num_proposals, 4).

        Returns:
            The tuple returned by RetinaTarget._build().
        """
        anchors_ph = tf.placeholder(tf.float32, [None, 4])
        gt_boxes_ph = tf.placeholder(tf.float32, [None, 5])
        retina_target_net = model(anchors_ph, gt_boxes_ph)
        with self.test_session() as sess:
            return sess.run(retina_target_net, feed_dict={
                anchors_ph: anchors,
                gt_boxes_ph: gt_boxes,
            })

    def testBasic(self):
        """Test a basic case.

        We have one ground truth box and three anchors. One should be
        background, one foreground, and one should be ignored.
        """
        config = EasyDict(self._config.copy())
        config['foreground_threshold'] = 0.5
        config['background_threshold_high'] = 0.4
        config['background_threshold_low'] = 0.0

        model = RetinaTarget(config, self._num_classes)

        gt_boxes = np.array(
            [(20, 20, 80, 100, self._placeholder_label)],
            dtype=np.float32
        )

        anchors = np.array(
            [
                (55, 75, 85, 105),  # Background
                                    # IoU ~0.1293
                (25, 21, 85, 105),  # Foreground
                                    # IoU ~0.7934
                (33, 34, 88, 129),  # Ignored
                                    # IoU ~0.4529
            ],
            dtype=np.float32
        )
        labels, _ = self._run_retina_target(model, gt_boxes, anchors)

        self.assertAllClose(
            labels,
            np.array([0., self._placeholder_label + 1., -1.]),
            atol=self._equality_delta
        )

    def testEmptyCase(self):
        """Test we're choosing the best box when none are above the
        foreground threshold.
        """
        gt_boxes = np.array(
            [(423, 30, 501, 80, self._placeholder_label)],
            dtype=np.float32
        )
        anchors = np.array(
            [
                (491, 70, 510, 92),  # IoU 0.0277
                (400, 60, 450, 92),  # IoU 0.1147
                (413, 40, 480, 77),  # IoU 0.4998: highest
                (411, 40, 480, 77),  # IoU 0.4914
            ],
            dtype=np.float32
        )

        labels, _ = self._run_retina_target(
            self._model, gt_boxes, anchors
        )

        # Assertions
        self.assertAlmostEqual(
            labels[2], self._placeholder_label + 1,
            delta=self._equality_delta
        )

        for i, label in enumerate(labels):
            if i != 2:
                # All other should be assigned either background (0) or
                # ignored (-1).
                self.assertLess(label, 1)

    def testAbsolutelyEmptyCase(self):
        """Test the case when there are no proposals with IoU > 0.
        """

        gt_boxes = np.array(
            [(40, 90, 100, 105, self._placeholder_label)],
            dtype=np.float32
        )

        anchors = np.array(
            [
                (0, 0, 39, 89),
                (101, 106, 300, 450),
                (340, 199, 410, 420),
            ],
            dtype=np.float32
        )

        labels, _ = self._run_retina_target(
            self._model, gt_boxes, anchors
        )

        foreground_number = labels[labels >= 1].shape[0]
        self.assertGreater(foreground_number, 0)

    def testManyGTBoxes(self):
        """Test a basic case with several gt boxes and several anchors.
        """
        anchors = np.array([
            # Foregrounds
            [0, 0, 10, 10],
            [0, 0, 10, 10],
            [10, 10, 20, 20],
            [10, 10, 20, 20],
            [20, 20, 30, 30],
            [20, 20, 30, 30],
            [30, 30, 40, 40],
            [30, 30, 40, 40],
            # Backgrounds
            [100, 100, 110, 110],
            [100, 100, 120, 120],
            [110, 110, 120, 120],
            [110, 110, 130, 130],
            [110, 110, 120, 120],
            [110, 110, 130, 130],
            [110, 110, 120, 120],
            [110, 110, 130, 130],
        ], dtype=np.float32)

        gt_boxes = np.array([
            [1, 1, 8, 8, self._placeholder_label],
            [11, 11, 18, 18, self._placeholder_label],
            [21, 21, 28, 28, self._placeholder_label],
            [31, 31, 38, 38, self._placeholder_label]
        ])
        labels, _ = self._run_retina_target(
            self._model, gt_boxes, anchors
        )

        # 8 foreground
        self.assertEqual(labels[labels >= 1].shape[0], 8)
        # 8 background or ignored
        self.assertEqual(labels[labels <= 0].shape[0], 8)

        # Check all 4 foregrounds are in the first 8 places of the label.
        self.assertTrue((labels.argsort()[-4:] < 8).all())


if __name__ == '__main__':
    tf.test.main()
