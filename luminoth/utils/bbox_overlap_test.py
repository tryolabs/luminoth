import numpy as np
import tensorflow as tf

from luminoth.utils.bbox_overlap import bbox_overlap_tf, bbox_overlap


class BBoxOverlapTest(tf.test.TestCase):
    """Tests for bbox_overlap
    bbox_overlap has a TensorFlow and a Numpy implementation.

    We test both at the same time by getting both values and making sure they
    are both equal before doing any assertions.
    """
    def tearDown(self):
        tf.reset_default_graph()

    def _get_iou(self, bbox1_val, bbox2_val):
        """Get IoU for two sets of bounding boxes.

        It also checks that both implementations return the same before
        returning.

        Args:
            bbox1_val: Array of shape (total_bbox1, 4).
            bbox2_val: Array of shape (total_bbox2, 4).

        Returns:
            iou: Array of shape (total_bbox1, total_bbox2)
        """
        bbox1 = tf.placeholder(tf.float32, (None, 4))
        bbox2 = tf.placeholder(tf.float32, (None, 4))
        iou = bbox_overlap_tf(bbox1, bbox2)

        with self.test_session() as sess:
            iou_val_tf = sess.run(iou, feed_dict={
                bbox1: np.array(bbox1_val),
                bbox2: np.array(bbox2_val),
            })

        iou_val_np = bbox_overlap(np.array(bbox1_val), np.array(bbox2_val))
        self.assertAllClose(iou_val_np, iou_val_tf)
        return iou_val_tf

    def testNoOverlap(self):
        # Single box test
        iou = self._get_iou([[0, 0, 10, 10]], [[11, 11, 20, 20]])
        self.assertAllEqual(iou, [[0.]])

        # Multiple boxes.
        iou = self._get_iou(
            [[0, 0, 10, 10], [5, 5, 10, 10]],
            [[11, 11, 20, 20], [15, 15, 20, 20]]
        )
        self.assertAllEqual(iou, [[0., 0.], [0., 0.]])

    def testAllOverlap(self):
        # Equal boxes
        iou = self._get_iou([[0, 0, 10, 10]], [[0, 0, 10, 10]])
        self.assertAllEqual(iou, [[1.]])

        # Crossed equal boxes.
        iou = self._get_iou(
            [[0, 0, 10, 10], [11, 11, 20, 20]],
            [[0, 0, 10, 10], [11, 11, 20, 20]]
        )
        # We should get an identity matrix.
        self.assertAllEqual(iou, [[1., 0.], [0., 1.]])

    def testInvalidBoxes(self):
        # Zero area, bbox1 has x_min == x_max
        iou = self._get_iou([[10, 0, 10, 10]], [[0, 0, 10, 10]])
        # self.assertAllEqual(iou, [[0.]]) TODO: Fails

        # Negative area, bbox1 has x_min > x_max (only by one)
        iou = self._get_iou([[10, 0, 9, 10]], [[0, 0, 10, 10]])
        self.assertAllEqual(iou, [[0.]])

        # Negative area, bbox1 has x_min > x_max
        iou = self._get_iou([[10, 0, 7, 10]], [[0, 0, 10, 10]])
        self.assertAllEqual(iou, [[0.]])

        # Negative area in both cases, both boxes equal but negative
        iou = self._get_iou([[10, 0, 7, 10]], [[10, 0, 7, 10]])
        self.assertAllEqual(iou, [[0.]])


if __name__ == '__main__':
    tf.test.main()
