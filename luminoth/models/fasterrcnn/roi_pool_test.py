import numpy as np
import tensorflow as tf

from easydict import EasyDict
from luminoth.models.fasterrcnn.roi_pool import ROIPoolingLayer


class ROIPoolingTest(tf.test.TestCase):

    def setUp(self):
        super(ROIPoolingTest, self).setUp()
        # Setup
        self.im_shape = (10, 10)
        self.config = EasyDict({
            'pooling_mode': 'crop',
            'pooled_width': 2,
            'pooled_height': 2,
            'padding': 'VALID',
        })

    def _run_roi_pooling(self, roi_proposals, pretrained, config):
        roi_proposals_tf = tf.placeholder(
            tf.float32, shape=roi_proposals.shape)
        pretrained_tf = tf.placeholder(tf.float32, shape=pretrained.shape)
        im_shape_tf = tf.placeholder(tf.float32, shape=(2,))

        model = ROIPoolingLayer(config, debug=True)
        results = model(roi_proposals_tf, pretrained_tf, im_shape_tf)

        with self.test_session() as sess:
            results = sess.run(results, feed_dict={
                roi_proposals_tf: roi_proposals,
                pretrained_tf: pretrained,
                im_shape_tf: self.im_shape,
            })
            return results

    def testBasic(self):
        """
        Test basic max pooling. We have 4 'roi_proposals' and use a kernel size
        of 2x2, then we will get as result a 'roi_pool' of 2x2.
        """
        roi_proposals = np.array([
            [0, 1, 1, 4, 4],  # Inside matA
            [1, 6, 1, 9, 4],  # Inside matB
            [2, 1, 6, 4, 9],  # Inside matC
            [3, 6, 6, 9, 9],  # Inside matD
        ])
        # Construct the pretrained map with four matrix.
        matA = np.ones((5, 5))
        matB = np.ones((5, 5)) + 1
        matC = np.ones((5, 5)) + 2
        matD = np.ones((5, 5)) + 3
        pretrained = np.bmat([[matA, matB], [matC, matD]])
        # Expand the dimensions to be compatible with ROIPoolingLayer.
        pretrained = np.expand_dims(pretrained, axis=0)
        pretrained = np.expand_dims(pretrained, axis=3)

        results = self._run_roi_pooling(roi_proposals, pretrained, self.config)
        print(results['crops'].shape)
        print(results['roi_pool'][0])

        # Check that crops has the correct shape.
        self.assertEqual(
            results['crops'].shape,
            (4, 4, 4, 1)
        )

        # Check that roi_pool has the correct shape.
        self.assertEqual(
            results['roi_pool'].shape,
            (4, 2, 2, 1)
        )

        # Check that max polling returns only 'one'
        self.assertAllEqual(
            results['roi_pool'][0],
            np.expand_dims(np.ones((2, 2)), axis=3)
        )

        # Check that max polling returns only 'two'
        self.assertAllEqual(
            results['roi_pool'][1],
            np.expand_dims(np.ones((2, 2)) + 1, axis=3)
        )

        # Check that max polling returns only 'three'
        self.assertAllEqual(
            results['roi_pool'][2],
            np.expand_dims(np.ones((2, 2)) + 2, axis=3)
        )

        # Check that max polling returns only 'four'
        self.assertAllEqual(
            results['roi_pool'][3],
            np.expand_dims(np.ones((2, 2)) + 3, axis=3)
        )


if __name__ == "__main__":
    tf.test.main()
