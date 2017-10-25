import tensorflow as tf
import numpy as np

from easydict import EasyDict

from luminoth.models.retina import Retina
from luminoth.models.retina.retina_proposal import RetinaProposal
from luminoth.utils.bbox_transform import encode
from luminoth.utils.config import get_base_config


class RetinaProposalTest(tf.test.TestCase):
    def setUp(self):
        super(RetinaProposalTest, self).setUp()
        self._config = EasyDict(get_base_config(Retina)['model']['proposal'])

        self._num_classes = 3
        self._placeholder_label = 2

        self._model = RetinaProposal(self._config, self._num_classes)

        self._equality_delta = 1.e-3

    def _run_retina_proposal(self, model, cls_prob, bbox_pred, anchors):
        cls_prob_ph = tf.placeholder(
            tf.float32, [None, model._num_classes + 1]
        )
        bbox_pred_ph = tf.placeholder(
            tf.float32, [None, 4]
        )
        anchors_ph = tf.placeholder(
            tf.float32, [None, 4]
        )
        net = model(cls_prob_ph, bbox_pred_ph, anchors_ph)
        with self.test_session() as sess:
            return sess.run(net, feed_dict={
                cls_prob_ph: cls_prob,
                bbox_pred_ph: bbox_pred,
                anchors_ph: anchors,
            })

    def testBackgroundFilter(self):
        """Tests that we're not returning an object when a proposal is background.

        This includes two sub-tests. One case in which there is a foreground
        proposal, and one in which all proposals are background. We use the
        same proposed_boxes and gt_boxes, but change the cls_prob.
        """
        config = EasyDict(self._config.copy())
        config['min_prob_threshold'] = 0.4
        model = RetinaProposal(config, self._num_classes)
        anchors = np.array([
            (85, 500, 730, 590),
            (50, 500, 70, 530),
            (700, 570, 740, 598),
        ])
        proposals = np.array([
            (101, 101, 201, 249),
            (200, 502, 209, 532),
            (86, 571, 743, 599),
        ])
        bbox_pred = encode(anchors, proposals)
        cls_prob_one_foreground = np.array([
            (0., .3, .25, .45),
            (1., 0., 0., 0.),
            (.3, .35, .2, .15),  # lower than the min_prob_threshold
        ])
        cls_prob_all_background = np.array([
            (.4, 0., .3, .3),
            (.8, .1, .1, 0.),
            (.7, .05, .2, .05),
        ])

        proposal_prediction_one_foreground = self._run_retina_proposal(
            model,
            cls_prob_one_foreground,
            bbox_pred,
            anchors,
        )
        proposal_prediction_all_background = self._run_retina_proposal(
            model,
            cls_prob_all_background,
            bbox_pred,
            anchors,
        )

        # Assertion for 'one foreground' case.
        self.assertEqual(
            proposal_prediction_one_foreground['objects'].shape[0],
            1
        )
        self.assertEqual(
            proposal_prediction_one_foreground['proposal_label'][0],
            2
        )
        # Assertion for 'all background' case.
        self.assertEqual(
            len(proposal_prediction_all_background['objects']), 0
        )

    def testLimits(self):
        """Tests that we're respecting the limits imposed by the config.
        """

        limits_config = EasyDict(self._config.copy())
        limits_config['min_prob_threshold'] = 0.2
        limits_config['class_max_detections'] = 2
        limits_config['total_max_detections'] = 3
        limits_config = EasyDict(limits_config)
        limits_num_classes = 2
        limits_model = RetinaProposal(limits_config, limits_num_classes)

        proposed_boxes = np.array([
            (0, 0, 1, 1),  # class 0
            (5, 5, 10, 10),  # class 1
            (15, 15, 20, 20),  # class 1
            (25, 25, 30, 30),  # class 0
            (35, 35, 40, 40),
            (38, 40, 65, 65),
            (70, 50, 90, 90),  # class 0
            (95, 95, 100, 100),
            (105, 105, 110, 110),  # class 1
        ])
        # All zeroes for our bbox_pred.
        bbox_pred = np.array([([0.] * 4)] * 9)
        cls_prob = np.array([
            (0., 1., 0.),
            (0., .2, .8),
            (0., .45, .55),
            (0., .55, .45),
            (1., 0., 0.),
            (1., 0., 0.),
            (0., .95, .05),
            (1., 0., 0.),
            (0., .495, .505),
        ])

        proposal_prediction = self._run_retina_proposal(
            limits_model,
            cls_prob,
            bbox_pred,
            proposed_boxes,
        )
        labels = proposal_prediction['proposal_label']
        num_class0 = labels[labels == 0].shape[0]
        num_class1 = labels[labels == 1].shape[0]

        self.assertLessEqual(num_class0, limits_config.class_max_detections)
        self.assertLessEqual(num_class1, limits_config.class_max_detections)
        num_total = labels.shape[0]
        self.assertLessEqual(num_total, limits_config.total_max_detections)


if __name__ == '__main__':
    tf.test.main()
