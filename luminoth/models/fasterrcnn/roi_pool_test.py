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
        # Construct the pretrained map with four matrix.
        self.multiplier_a = 1
        self.multiplier_b = 2
        self.multiplier_c = 3
        self.multiplier_d = 4
        mat_a = np.ones((5, 5)) * self.multiplier_a
        mat_b = np.ones((5, 5)) * self.multiplier_b
        mat_c = np.ones((5, 5)) * self.multiplier_c
        mat_d = np.ones((5, 5)) * self.multiplier_d
        self.pretrained = np.bmat([[mat_a, mat_b], [mat_c, mat_d]])
        # Expand the dimensions to be compatible with ROIPoolingLayer.
        self.pretrained = np.expand_dims(self.pretrained, axis=0)
        self.pretrained = np.expand_dims(self.pretrained, axis=3)
        # pretrained:
        #           mat_a | mat_b
        #           -------------
        #           mat_c | mat_d
        tf.reset_default_graph()

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
        Test basic max pooling. We have 4 'roi_proposals' and use a 'pool size'
        of 2x2 ('pooled_width': 2, 'pooled_height': 2), then we will get as
        result a 'roi_pool' of 2x2.
        """
        roi_proposals = np.array([
            [1, 1, 4, 4],  # Inside mat_A
            [6, 1, 9, 4],  # Inside mat_B
            [1, 6, 4, 9],  # Inside mat_C
            [6, 6, 9, 9],  # Inside mat_D
        ])

        results = self._run_roi_pooling(
            roi_proposals, self.pretrained, self.config)

        # Check that crops has the correct shape. This is (4, 4, 4, 1)
        # because we have 4 proposals, 'pool size' = 2x2 then the
        # tf.image.crop_and_resize function duplicates that size.
        self.assertEqual(
            results['crops'].shape,
            (4, 4, 4, 1)
        )

        # Check that roi_pool has the correct shape. This is (4, 2, 2, 1)
        # because we have 4 proposals, 'pool size' = 2x2.
        self.assertEqual(
            results['roi_pool'].shape,
            (4, 2, 2, 1)
        )

        results['roi_pool'] = np.squeeze(results['roi_pool'], axis=3)
        # Check that max polling returns only 'multiplier_a'
        self.assertAllEqual(
            results['roi_pool'][0],
            np.ones((2, 2)) * self.multiplier_a
        )

        # Check that max polling returns only 'multiplier_b'
        self.assertAllEqual(
            results['roi_pool'][1],
            np.ones((2, 2)) * self.multiplier_b
        )

        # Check that max polling returns only 'multiplier_c'
        self.assertAllEqual(
            results['roi_pool'][2],
            np.ones((2, 2)) * self.multiplier_c
        )

        # Check that max polling returns only 'multiplier_d'
        self.assertAllEqual(
            results['roi_pool'][3],
            np.ones((2, 2)) * self.multiplier_d
        )

    def testMaxPoolingWithoutInterpolation(self):
        """
        Test max pooling with a little bit more complex 'roi_proposals'.
        We have 4 'roi_proposals' and use a 'pool size'
        of 2x2 ('pooled_width': 2, 'pooled_height': 2), then we will get as
        result a 'roi_pool' of 2x2. with
        """
        roi_proposals = np.array([
            [3, 1, 6, 4],  # Across mat_A and mat_B (half-half)
            [1, 3, 4, 7],  # Across mat_A and mat_C (half-half)
            [5, 3, 9, 7],  # Inside mat_B and mat_D (half-half)
            [3, 6, 6, 9],  # Inside mat_C and mat_D (half-half)
        ])
        a = self.multiplier_a
        b = self.multiplier_b
        c = self.multiplier_c
        d = self.multiplier_d

        results = self._run_roi_pooling(
            roi_proposals, self.pretrained, self.config)

        # Check that crops has the correct shape. This is (4, 4, 4, 1)
        # because we have 4 proposals, 'pool size' = 2x2 then the
        # tf.image.crop_and_resize function duplicates that size.
        self.assertEqual(
            results['crops'].shape,
            (4, 4, 4, 1)
        )

        # Check that roi_pool has the correct shape. This is (4, 2, 2, 1)
        # because we have 4 proposals, 'pool size' = 2x2.
        self.assertEqual(
            results['roi_pool'].shape,
            (4, 2, 2, 1)
        )

        results['roi_pool'] = np.squeeze(results['roi_pool'], axis=3)
        # Check that max polling returns a column of 'one' and
        # a column of 'two'.
        self.assertAllEqual(
            results['roi_pool'][0],
            np.array([[a, b], [a, b]])
        )

        # Check that max polling returns a row of 'one' and
        # a row of 'three'.
        self.assertAllEqual(
            results['roi_pool'][1],
            np.array([[a, a], [c, c]])
        )

        # Check that max polling returns a row of 'one' and
        # a row of 'three'.
        self.assertAllEqual(
            results['roi_pool'][2],
            np.array([[b, b], [d, d]])
        )

        # Check that max polling returns a column of 'three' and
        # a column of 'four'.
        self.assertAllEqual(
            results['roi_pool'][3],
            np.array([[c, d], [c, d]])
        )

    def testMaxPoolingWithInterpolation(self):
        """
        Test max pooling with bilinear interpolation.
        We have 4 'roi_proposals' and use a 'pool size'
        of 2x2 ('pooled_width': 2, 'pooled_height': 2), then we will get as
        result a 'roi_pool' of 2x2.
        """

        roi_proposals = np.array([
            [4, 1, 7, 4],  # Across mat_A and mat_B (1/4 - 3/4)
            [1, 4, 4, 8],  # Across mat_A and mat_C (1/4 - 3/4)
            [5, 4, 9, 8],  # Inside mat_B and mat_D (1/4 - 3/4)
            [4, 6, 7, 9],  # Inside mat_C and mat_D (1/4 - 3/4)
        ])
        a = self.multiplier_a
        b = self.multiplier_b
        c = self.multiplier_c
        d = self.multiplier_d

        results = self._run_roi_pooling(
            roi_proposals, self.pretrained, self.config)

        results['roi_pool'] = np.squeeze(results['roi_pool'], axis=3)

        # Check that max polling returns values greater or equal
        # than 'a' and crops returns values lower or equal than 'b'
        self.assertTrue(
            np.greater_equal(results['roi_pool'][0], a).all()
        )

        self.assertTrue(
            np.less_equal(results['crops'][0], b).all()
        )

        # Check that max polling returns values greater or equal
        # than 'a' and crops returns values lower or equal than 'c'
        self.assertTrue(
            np.greater_equal(results['roi_pool'][1], a).all()
        )

        self.assertTrue(
            np.less_equal(results['crops'][1], c).all()
        )

        # Check that max polling returns values greater or equal
        # than 'b' and crops returns values lower or equal than 'd'
        self.assertTrue(
            np.greater_equal(results['roi_pool'][2], b).all()
        )

        self.assertTrue(
            np.less_equal(results['crops'][2], d).all()
        )

        # Check that max polling returns values greater or equal
        # than 'c' and crops returns values lower or equal than 'd'
        self.assertTrue(
            np.greater_equal(results['roi_pool'][3], c).all()
        )

        self.assertTrue(
            np.less_equal(results['crops'][3], d).all()
        )


if __name__ == "__main__":
    tf.test.main()
