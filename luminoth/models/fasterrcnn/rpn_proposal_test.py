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
            'clip_after_nms': False,
            'filter_outside_anchors': False,
            'apply_nms': True,
            'min_prob_threshold': 0.0,
        })
        tf.reset_default_graph()

    def _run_rpn_proposal(self, all_anchors, rpn_cls_prob, config,
                          gt_boxes=None, rpn_bbox_pred=None):
        """
        Define one of gt_boxes or rpn_bbox_pred.

        If using gt_boxes, the correct rpn_bbox_pred for those gt_boxes will
        be used.
        """
        feed_dict = {}
        rpn_cls_prob_tf = tf.placeholder(
            tf.float32, shape=(all_anchors.shape[0], 2))
        feed_dict[rpn_cls_prob_tf] = rpn_cls_prob
        im_size_tf = tf.placeholder(tf.float32, shape=(2,))
        feed_dict[im_size_tf] = self.im_size
        all_anchors_tf = tf.placeholder(tf.float32, shape=all_anchors.shape)
        feed_dict[all_anchors_tf] = all_anchors
        if rpn_bbox_pred is None and gt_boxes is not None:
            # Here we encode 'all_anchors' and 'gt_boxes' to get corrects
            # predictions that RPNProposal can decodes.
            rpn_bbox_pred_tf = encode(all_anchors, gt_boxes)
        else:
            rpn_bbox_pred_tf = tf.placeholder(
                tf.float32, shape=rpn_bbox_pred.shape
            )
            feed_dict[rpn_bbox_pred_tf] = rpn_bbox_pred

        model = RPNProposal(all_anchors.shape[0], config, debug=True)
        results = model(
            rpn_cls_prob_tf, rpn_bbox_pred_tf, all_anchors_tf, im_size_tf)

        with self.test_session() as sess:
            results = sess.run(results, feed_dict=feed_dict)
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
            all_anchors, rpn_cls_prob, self.config, gt_boxes=gt_boxes)

        # Check we get exactly 2 'nms proposals' because 2 IoU equals to 0.
        # Also check that we get the corrects scores.
        self.assertEqual(
            results['proposals'].shape,
            (2, 4)
        )

        self.assertAllClose(
            results['scores'],
            [0.9, 0.8]
        )

        config['nms_threshold'] = 0.3

        results = self._run_rpn_proposal(
            all_anchors, rpn_cls_prob, self.config, gt_boxes=gt_boxes)

        # Check we get exactly 3 'nms proposals' because 3 IoU lowers than 0.3.
        # Also check that we get the corrects scores.
        self.assertEqual(
            results['proposals'].shape,
            (3, 4)
        )

        self.assertAllClose(
            results['scores'],
            [0.9, 0.8, 0.2]
        )

        config['nms_threshold'] = 0.6

        results = self._run_rpn_proposal(
            all_anchors, rpn_cls_prob, self.config, gt_boxes=gt_boxes)

        # Check we get exactly 3 'nms proposals' because 3 IoU lowers than 0.3.
        # Also check that we get the corrects scores.
        self.assertEqual(
            results['proposals'].shape,
            (3, 4)
        )

        self.assertAllClose(
            results['scores'],
            [0.9, 0.8, 0.2]
        )

        config['nms_threshold'] = 0.8

        results = self._run_rpn_proposal(
            all_anchors, rpn_cls_prob, self.config, gt_boxes=gt_boxes)

        # Check we get exactly 3 'nms proposals' because 3 IoU lowers than 0.8.
        # Also check that we get the corrects scores.
        self.assertEqual(
            results['proposals'].shape,
            (3, 4)
        )

        self.assertAllClose(
            results['scores'],
            [0.9, 0.8, 0.2]
        )

        config['nms_threshold'] = 1.0

        results = self._run_rpn_proposal(
            all_anchors, rpn_cls_prob, self.config, gt_boxes=gt_boxes)

        # Check we get 'post_nms_top_n' nms proposals because
        # 'nms_threshold' = 1 and this only removes duplicates.
        self.assertEqual(
            results['proposals'].shape,
            (4, 4)
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
            all_anchors, rpn_cls_prob, self.config, gt_boxes=gt_boxes)

        # Check we get exactly 3 'nms proposals' and 3 'unsorted_proposals'
        # because we have 4 gt_boxes, but 1 outsider (and nms_threshold = 1).
        self.assertEqual(
            results['proposals'].shape,
            (3, 4)
        )

        # We don't remove proposals outside, we just clip them.
        self.assertEqual(
            results['unsorted_proposals'].shape,
            (4, 4)
        )

        # Also check that we get the corrects scores.
        self.assertAllClose(
            results['scores'],
            [0.7, 0.6, 0.2]
        )

        config = self.config
        config['post_nms_top_n'] = 2

        results = self._run_rpn_proposal(
            all_anchors, rpn_cls_prob, config, gt_boxes=gt_boxes)

        # Check that with a post_nms_top_n = 2 we have only 2 'nms proposals'
        # but 3 'unsorted_proposals'.
        self.assertAllEqual(
            results['proposals'].shape,
            (2, 4)
        )

        self.assertEqual(
            results['unsorted_proposals'].shape,
            (4, 4)
        )

        # Also check that we get the corrects scores.
        self.assertAllClose(
            results['scores'],
            [0.7, 0.6]
        )

        # Sorted
        self.assertAllClose(
            results['sorted_top_scores'],
            [0.7, 0.6, 0.2, 0.1]
        )

        # Check that we only filter by pre_nms_top_n
        config['post_nms_top_n'] = 3
        config['pre_nms_top_n'] = 2

        results = self._run_rpn_proposal(
            all_anchors, rpn_cls_prob, config, gt_boxes=gt_boxes)

        # Check that with a post_nms_top_n = 3 and pre_nms_top = 2
        # we have only 2 'nms proposals' and 2 'unsorted_proposals'.

        self.assertAllEqual(
            results['proposals'].shape,
            (2, 4)
        )

        # Filter pre nms
        self.assertEqual(
            results['sorted_top_proposals'].shape,
            (2, 4)
        )

        # Also check that we get the corrects scores.
        self.assertAllClose(
            results['scores'],
            [0.7, 0.6]
        )

        self.assertAllClose(
            results['sorted_top_scores'],
            [0.7, 0.6]
        )

        config['post_nms_top_n'] = 1
        config['pre_nms_top_n'] = 2

        results = self._run_rpn_proposal(
            all_anchors, rpn_cls_prob, config, gt_boxes=gt_boxes)

        # Check that with a post_nms_top_n = 1 and pre_nms_top = 2
        # we have only 1 'nms proposals' and 2 'unsorted_proposals'.
        self.assertAllEqual(
            results['proposals'].shape,
            (1, 4)
        )

        self.assertEqual(
            results['sorted_top_proposals'].shape,
            (2, 4)
        )

        # Also check that we get the corrects scores.
        self.assertAllClose(
            results['scores'],
            [0.7]
        )

        self.assertAllClose(
            results['sorted_top_scores'],
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
            all_anchors, rpn_cls_prob, self.config, gt_boxes=gt_boxes)

        # Check we get exactly 2 'proposals' and 2 'unsorted_proposals' because
        # we have 4 gt_boxes, but 2 with negative area (and nms_threshold = 1).
        self.assertEqual(
            results['proposals'].shape,
            (2, 4)
        )

        self.assertEqual(
            results['unsorted_proposals'].shape,
            (2, 4)
        )

    def testNegativeAreaProposals(self):
        all_anchors = np.array([
            [11, 13, 12, 16],
            [10, 10, 9, 9],  # invalid anchor will transform to an invalid
            [11, 13, 12, 28],  # proposal. we are cheating here but it's almost
            [7, 13, 34, 30],  # the same.
        ])
        rpn_cls_prob = np.array([
            [0.3, 0.7],
            [0.4, 0.6],
            [0.9, 0.1],
            [0.8, 0.2]
        ])
        rpn_bbox_pred = np.array([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ])

        results = self._run_rpn_proposal(
            all_anchors, rpn_cls_prob, self.config,
            rpn_bbox_pred=rpn_bbox_pred
        )

        self.assertEqual(
            results['unsorted_proposals'].shape,
            (3, 4)
        )

    def testClippingOfProposals(self):
        """
        Test clipping of proposals before and after NMS
        """
        # Before NMS
        gt_boxes = np.array([
            [0, 0, 10, 12],
            [10, 10, 20, 22],
            [10, 10, 20, 22],
            [30, 25, 39, 39],
        ])
        all_anchors = np.array([
            [-20, -10, 12, 6],
            [2, -10, 20, 20],
            [0, 0, 12, 16],
            [2, -10, 20, 2],
        ])
        rpn_cls_prob = np.array([
            [0.3, 0.7],
            [0.4, 0.6],
            [0.3, 0.7],
            [0.1, 0.9],
        ])

        rpn_bbox_pred = np.array([  # This is set to zeros so when decode is
            [0, 0, 0, 0],           # applied in RPNProposal the anchors don't
            [0, 0, 0, 0],           # change, leaving us with unclipped
            [0, 0, 0, 0],           # proposals.
            [0, 0, 0, 0],
        ])
        config = EasyDict(self.config)
        config['clip_after_nms'] = False
        results_before = self._run_rpn_proposal(
            all_anchors, rpn_cls_prob, config, gt_boxes=gt_boxes,
            rpn_bbox_pred=rpn_bbox_pred)
        im_size = tf.placeholder(tf.float32, shape=(2,))
        proposals_unclipped = tf.placeholder(
            tf.float32, shape=(results_before['proposals_unclipped'].shape))
        clip_bboxes_tf = clip_boxes(proposals_unclipped, im_size)

        with self.test_session() as sess:
            clipped_proposals = sess.run(clip_bboxes_tf, feed_dict={
                proposals_unclipped: results_before['proposals_unclipped'],
                im_size: self.im_size
            })

        # Check we clip proposals right after filtering the invalid area ones.
        self.assertAllEqual(
            results_before['unsorted_proposals'],
            clipped_proposals
        )

        # Checks all NMS proposals have values inside the image boundaries
        proposals = results_before['proposals']
        self.assertTrue((proposals >= 0).all())
        self.assertTrue(
            (proposals < np.array(self.im_size + self.im_size)).all()
        )

        # After NMS
        config['clip_after_nms'] = True
        results_after = self._run_rpn_proposal(
            all_anchors, rpn_cls_prob, config, gt_boxes=gt_boxes,
            rpn_bbox_pred=rpn_bbox_pred)
        im_size = tf.placeholder(tf.float32, shape=(2,))
        proposals_unclipped = tf.placeholder(
            tf.float32, shape=(results_after['proposals_unclipped'].shape))
        clip_bboxes_tf = clip_boxes(proposals_unclipped, im_size)

        with self.test_session() as sess:
            clipped_proposals = sess.run(clip_bboxes_tf, feed_dict={
                proposals_unclipped: results_after['proposals_unclipped'],
                im_size: self.im_size
            })

        # Check we don't clip proposals in the beginning of the function.
        self.assertAllEqual(
            results_after['unsorted_proposals'],
            results_after['proposals_unclipped']
        )

        proposals = results_after['proposals']
        # Checks all NMS proposals have values inside the image boundaries
        self.assertTrue((proposals >= 0).all())
        self.assertTrue(
            (proposals < np.array(self.im_size + self.im_size)).all()
        )

    def testFilterOutsideAnchors(self):
        """
        Test clipping of proposals before and after NMS
        """
        gt_boxes = np.array([
            [0, 0, 10, 12],
            [10, 10, 20, 22],
            [10, 10, 20, 22],
            [30, 25, 39, 39],
            [30, 25, 39, 39],
        ])
        all_anchors = np.array([    # Img_size (40, 40)
            [-20, -10, 12, 6],      # Should be filtered
            [2, 10, 20, 20],        # Valid anchor
            [0, 0, 50, 16],         # Should be filtered
            [2, -10, 20, 50],       # Should be filtered
            [25, 30, 27, 33],       # Valid anchor
        ])
        rpn_cls_prob = np.array([
            [0.3, 0.7],
            [0.4, 0.6],
            [0.3, 0.7],
            [0.1, 0.9],
            [0.2, 0.8],
        ])
        config = EasyDict(self.config)
        config['filter_outside_anchors'] = False
        results_without_filter = self._run_rpn_proposal(
            all_anchors, rpn_cls_prob, config, gt_boxes=gt_boxes)

        # Check that all_proposals contain the outside anchors
        self.assertAllEqual(
            results_without_filter['all_proposals'].shape,
            all_anchors.shape)

        config['filter_outside_anchors'] = True
        results_with_filter = self._run_rpn_proposal(
            all_anchors, rpn_cls_prob, config, gt_boxes=gt_boxes)
        self.assertAllEqual(
            results_with_filter['all_proposals'].shape,
            (2, 4))


if __name__ == "__main__":
    tf.test.main()
