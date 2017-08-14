import numpy as np
import tensorflow as tf

from easydict import EasyDict
from luminoth.models.fasterrcnn.rpn_proposal import RPNProposal
from luminoth.utils.bbox_transform_tf import encode, clip_boxes


class RPNProposalTest(tf.test.TestCase):

    def setUp(self):
        super(RPNProposalTest, self).setUp()
        # Setup
        self.im_size = (40, 40)
        self.config = EasyDict({
            'pre_nms_top_n': 4,
            'post_nms_top_n': 3,
            'nms_threshold': 1,
            'min_size': 0,
        })

    def _run_rpn_proposal(self, all_anchors, gt_boxes, rpn_cls_prob, config):
        rpn_cls_prob_tf = tf.placeholder(
            tf.float32, shape=(all_anchors.shape[0], 2))
        im_size_tf = tf.placeholder(tf.float32, shape=(2,))
        all_anchors_tf = tf.placeholder(tf.float32, shape=all_anchors.shape)
        # Here we encode 'all_anchors' and 'gt_boxes' to get corrects
        # predictions that RPNProposal can decodes.
        rpn_bbox_pred = encode(all_anchors, gt_boxes)

        model = RPNProposal(all_anchors.shape[0], config)
        results = model(
            rpn_cls_prob_tf, rpn_bbox_pred, all_anchors_tf, im_size_tf)

        with self.test_session() as sess:
            results = sess.run(results, feed_dict={
                rpn_cls_prob_tf: rpn_cls_prob,
                all_anchors_tf: all_anchors,
                im_size_tf: self.im_size,
            })
            return results

    def testNMSThreshold(self):
        """
        Test nms threshold
        """
        gt_boxes = np.array([
            [10, 10, 26, 36],
            [10, 10, 20, 22],
            [10, 11, 20, 21],
            [19, 30, 33, 38],
        ])
        """
        IoU Matrix of gt_boxes
        [[ 1.          0.31154684  0.26361656  0.10408922]
         [ 0.31154684  1.          0.84615385  0.        ]
         [ 0.26361656  0.84615385  1.          0.        ]
         [ 0.10408922  0.          0.          1.        ]]
        """
        all_anchors = np.array([
            [11, 13, 34, 31],
            [10, 10, 20, 22],
            [11, 13, 34, 28],
            [21, 29, 34, 37],
        ])
        rpn_cls_prob = np.array([
            [0.8, 0.2],
            [0.1, 0.9],
            [0.4, 0.6],
            [0.2, 0.8]
        ])
        config = self.config
        config['post_nms_top_n'] = 4
        config['nms_threshold'] = 0.0

        results = self._run_rpn_proposal(
            all_anchors, gt_boxes, rpn_cls_prob, self.config)

        # Check we get exactly 2 'nms proposals' because 2 IoU equals to 0.
        # Also check that we get the corrects scores.
        self.assertEqual(
            results['nms_proposals'].shape,
            (2, 5)
        )

        self.assertAllClose(
            results['nms_proposals_scores'],
            [0.9, 0.8]
        )

        config['nms_threshold'] = 0.3

        results = self._run_rpn_proposal(
            all_anchors, gt_boxes, rpn_cls_prob, self.config)

        # Check we get exactly 3 'nms proposals' because 3 IoU lowers than 0.3.
        # Also check that we get the corrects scores.
        self.assertEqual(
            results['nms_proposals'].shape,
            (3, 5)
        )

        self.assertAllClose(
            results['nms_proposals_scores'],
            [0.9, 0.8, 0.2]
        )

        config['nms_threshold'] = 0.6

        results = self._run_rpn_proposal(
            all_anchors, gt_boxes, rpn_cls_prob, self.config)

        # Check we get exactly 3 'nms proposals' because 3 IoU lowers than 0.3.
        # Also check that we get the corrects scores.
        self.assertEqual(
            results['nms_proposals'].shape,
            (3, 5)
        )

        self.assertAllClose(
            results['nms_proposals_scores'],
            [0.9, 0.8, 0.2]
        )

        config['nms_threshold'] = 0.8

        results = self._run_rpn_proposal(
            all_anchors, gt_boxes, rpn_cls_prob, self.config)

        # Check we get exactly 3 'nms proposals' because 3 IoU lowers than 0.8.
        # Also check that we get the corrects scores.
        self.assertEqual(
            results['nms_proposals'].shape,
            (3, 5)
        )

        self.assertAllClose(
            results['nms_proposals_scores'],
            [0.9, 0.8, 0.2]
        )

        config['nms_threshold'] = 1.0

        results = self._run_rpn_proposal(
            all_anchors, gt_boxes, rpn_cls_prob, self.config)

        # Check we get 'post_nms_top_n' nms proposals because
        # 'nms_threshold' = 1 and this only removes duplicates.
        self.assertEqual(
            results['nms_proposals'].shape,
            (4, 5)
        )

    def testOutsidersAndTopN(self):
        """
        Test outside anchors and topN filters
        """
        gt_boxes = np.array([
            [10, 10, 20, 22],
            [10, 10, 20, 22],
            [10, 10, 20, 50],  # Outside anchor
            [10, 10, 20, 22],
        ])
        all_anchors = np.array([
            [11, 13, 34, 31],
            [10, 10, 20, 22],
            [11, 13, 34, 40],
            [7, 13, 34, 30],
        ])
        rpn_cls_prob = np.array([
            [0.3, 0.7],
            [0.4, 0.6],
            [0.9, 0.1],
            [0.8, 0.2]
        ])

        results = self._run_rpn_proposal(
            all_anchors, gt_boxes, rpn_cls_prob, self.config)

        # Check we get exactly 3 'nms proposals' and 3 'proposals' because
        # we have 4 gt_boxes, but 1 outsider (and nms_threshold = 1).
        self.assertEqual(
            results['nms_proposals'].shape,
            (3, 5)
        )

        self.assertEqual(
            results['proposals'].shape,
            (3, 4)
        )

        # Also check that we get the corrects scores.
        self.assertAllClose(
            results['nms_proposals_scores'],
            [0.7, 0.6, 0.2]
        )

        config = self.config
        config['post_nms_top_n'] = 2

        results = self._run_rpn_proposal(
            all_anchors, gt_boxes, rpn_cls_prob, config)

        # Check that with a post_nms_top_n = 2 we have only 2 'nms proposals'
        # but 3 'proposals'.
        self.assertAllEqual(
            results['nms_proposals'].shape,
            (2, 5)
        )

        self.assertEqual(
            results['proposals'].shape,
            (3, 4)
        )

        # Also check that we get the corrects scores.
        self.assertAllClose(
            results['nms_proposals_scores'],
            [0.7, 0.6]
        )

        self.assertAllClose(
            results['scores'],
            [0.7, 0.6, 0.2]
        )

        # Check that we only filter by pre_nms_top_n
        config['post_nms_top_n'] = 3
        config['pre_nms_top_n'] = 2

        results = self._run_rpn_proposal(
            all_anchors, gt_boxes, rpn_cls_prob, config)

        # Check that with a post_nms_top_n = 3 and pre_nms_top = 2
        # we have only 2 'nms proposals' and 2 'proposals'.

        self.assertAllEqual(
            results['nms_proposals'].shape,
            (2, 5)
        )

        self.assertEqual(
            results['proposals'].shape,
            (2, 4)
        )

        # Also check that we get the corrects scores.
        self.assertAllClose(
            results['nms_proposals_scores'],
            [0.7, 0.6]
        )

        self.assertAllClose(
            results['scores'],
            [0.7, 0.6]
        )

        config['post_nms_top_n'] = 1
        config['pre_nms_top_n'] = 2

        results = self._run_rpn_proposal(
            all_anchors, gt_boxes, rpn_cls_prob, config)

        # Check that with a post_nms_top_n = 1 and pre_nms_top = 2
        # we have only 1 'nms proposals' and 2 'proposals'.
        self.assertAllEqual(
            results['nms_proposals'].shape,
            (1, 5)
        )

        self.assertEqual(
            results['proposals'].shape,
            (2, 4)
        )

        # Also check that we get the corrects scores.
        self.assertAllClose(
            results['nms_proposals_scores'],
            [0.7]
        )

        self.assertAllClose(
            results['scores'],
            [0.7, 0.6]
        )

    def testNegativeArea(self):
        """
        Test negative area filters
        """
        gt_boxes = np.array([
            [10, 10, 20, 3],  # Negative area
            [10, 10, 20, 22],
            [10, 10, 8, 22],  # Negative area
            [10, 10, 20, 22],
        ])
        all_anchors = np.array([
            [11, 13, 12, 16],
            [10, 10, 20, 22],
            [11, 13, 12, 19],
            [7, 13, 34, 30],
        ])
        rpn_cls_prob = np.array([
            [0.3, 0.7],
            [0.4, 0.6],
            [0.9, 0.1],
            [0.8, 0.2]
        ])

        results = self._run_rpn_proposal(
            all_anchors, gt_boxes, rpn_cls_prob, self.config)

        # Check we get exactly 2 'nms proposals' and 2 'proposals' because
        # we have 4 gt_boxes, but 2 with negative area (and nms_threshold = 1).
        self.assertEqual(
            results['nms_proposals'].shape,
            (2, 5)
        )

        self.assertEqual(
            results['proposals'].shape,
            (2, 4)
        )

    def testClippingOfProposals(self):
        """
        Test clipping of proposals
        """
        gt_boxes = np.array([
            [10, 10, 20, 22],
            [10, 10, 20, 22],
            [10, 10, 20, 22],
            [10, 10, 20, 22],
        ])
        all_anchors = np.array([
            [11, 13, 12, 16],
            [10, 10, 20, 22],
            [11, 13, 12, 28],
            [7, 13, 34, 30],
        ])
        rpn_cls_prob = np.array([
            [0.3, 0.7],
            [0.4, 0.6],
            [0.9, 0.1],
            [0.8, 0.2]
        ])

        results = self._run_rpn_proposal(
            all_anchors, gt_boxes, rpn_cls_prob, self.config)

        im_size = tf.placeholder(tf.float32, shape=(2,))
        proposals = tf.placeholder(
            tf.float32, shape=(results['proposals'].shape))
        clip_bboxes_tf = clip_boxes(proposals, im_size)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            clipped_proposals = sess.run(clip_bboxes_tf, feed_dict={
                proposals: results['proposals'],
                im_size: self.im_size
            })

        # Check we get proposals clipped to the image.
        self.assertAllEqual(
            results['proposals'],
            clipped_proposals
        )


if __name__ == "__main__":
    tf.test.main()
