import numpy as np
import tensorflow as tf

from luminoth.utils.bbox_transform import (
    encode as encode_np, decode as decode_np, clip_boxes as clip_boxes_np
)
from luminoth.utils.bbox_transform_tf import (
    encode as encode_tf, decode as decode_tf, clip_boxes as clip_boxes_tf
)
from luminoth.utils.test.gt_boxes import generate_gt_boxes


class BBoxTransformTest(tf.test.TestCase):
    def tearDown(self):
        tf.reset_default_graph()

    def _encode(self, proposals, gt_boxes):
        """
        Encodes the adjustment from proposals to GT boxes using both the
        TensorFlow and the Numpy implementation.

        Asserts that both results are equal.
        """
        proposals_tf = tf.placeholder(tf.float32, shape=proposals.shape)
        gt_boxes_tf = tf.placeholder(tf.float32, shape=gt_boxes.shape)

        encoded_tf = encode_tf(proposals_tf, gt_boxes_tf)
        with self.test_session() as sess:
            encoded_tf_val = sess.run(encoded_tf, feed_dict={
                proposals_tf: proposals,
                gt_boxes_tf: gt_boxes,
            })

        encoded_np = encode_np(proposals, gt_boxes)

        self.assertAllClose(encoded_np, encoded_tf_val)
        return encoded_np

    def _decode(self, proposals, deltas):
        """
        Encodes the final boxes from proposals with deltas, using both the
        TensorFlow and the Numpy implementation.

        Asserts that both results are equal.
        """
        proposals_tf = tf.placeholder(tf.float32, shape=proposals.shape)
        deltas_tf = tf.placeholder(tf.float32, shape=deltas.shape)

        decoded_tf = decode_tf(proposals_tf, deltas_tf)
        with self.test_session() as sess:
            decoded_tf_val = sess.run(decoded_tf, feed_dict={
                proposals_tf: proposals,
                deltas_tf: deltas,
            })

        decoded_np = decode_np(proposals, deltas)

        self.assertAllClose(decoded_np, decoded_tf_val)
        return decoded_np

    def _encode_decode(self, proposals, gt_boxes):
        """
        Encode and decode to check inverse.
        """
        proposals_tf = tf.placeholder(tf.float32, shape=proposals.shape)
        gt_boxes_tf = tf.placeholder(tf.float32, shape=gt_boxes.shape)
        deltas_tf = encode_tf(proposals_tf, gt_boxes_tf)
        decoded_gt_boxes_tf = decode_tf(proposals_tf, deltas_tf)

        with self.test_session() as sess:
            decoded_gt_boxes = sess.run(decoded_gt_boxes_tf, feed_dict={
                proposals_tf: proposals,
                gt_boxes_tf: gt_boxes,
            })
            self.assertAllClose(decoded_gt_boxes, gt_boxes, atol=1e-04)

    def _clip_boxes(self, proposals, image_shape):
        """
        Clips boxes to image shape using both the TensorFlow and the Numpy
        implementation.

        Asserts that both results are equal.
        """
        proposals_tf = tf.placeholder(tf.float32, shape=proposals.shape)
        image_shape_tf = tf.placeholder(tf.int32, shape=(2,))
        clipped_tf = clip_boxes_tf(proposals, image_shape_tf)
        with self.test_session() as sess:
            clipped_tf_val = sess.run(clipped_tf, feed_dict={
                proposals_tf: proposals,
                image_shape_tf: image_shape,
            })

        clipped_np_val = clip_boxes_np(proposals, image_shape)
        self.assertAllClose(clipped_np_val, clipped_tf_val)
        return clipped_np_val

    def testEncodeDecode(self):
        # 1 vs 1 already equal encode and decode.
        proposal = generate_gt_boxes(1, image_size=100)
        gt_boxes = proposal

        deltas = self._encode(proposal, gt_boxes)
        decoded_gt_boxes = self._decode(proposal, deltas)

        self.assertAllEqual(deltas, np.zeros((1, 4)))
        self.assertAllClose(gt_boxes, decoded_gt_boxes)

        # 3 vs 3 already equal encode and decode
        proposal = generate_gt_boxes(3, image_size=100)
        gt_boxes = proposal

        deltas = self._encode(proposal, gt_boxes)
        decoded_gt_boxes = self._decode(proposal, deltas)

        self.assertAllEqual(deltas, np.zeros((3, 4)))
        self.assertAllClose(gt_boxes, decoded_gt_boxes)

        # 3 vs 4 different encode and decode
        proposal = generate_gt_boxes(3, image_size=100)
        gt_boxes = generate_gt_boxes(3, image_size=100)

        deltas = self._encode(proposal, gt_boxes)
        decoded_gt_boxes = self._decode(proposal, deltas)

        self.assertAllEqual(deltas.shape, (3, 4))
        self.assertAllClose(gt_boxes, decoded_gt_boxes)

    def testClipBboxes(self):
        image_shape = (50, 60)  # height, width
        boxes = np.array([
            [-1, 10, 20, 20],  # x_min is left of the image.
            [10, -1, 20, 20],  # y_min is above the image.
            [10, 10, 60, 20],  # x_max is right of the image.
            [10, 10, 20, 50],  # y_max is below the image.
            [10, 10, 20, 20],  # everything is in place
            [60, 50, 60, 50],  # complete box is outside the image.
        ])
        clipped_bboxes = self._clip_boxes(boxes, image_shape)
        self.assertAllEqual(clipped_bboxes, [
            [0, 10, 20, 20],
            [10, 0, 20, 20],
            [10, 10, 59, 20],
            [10, 10, 20, 49],
            [10, 10, 20, 20],
            [59, 49, 59, 49],
        ])

    def testEncodeDecodeRandomizedValues(self):
        for i in range(1, 2000, 117):
            gt_boxes = generate_gt_boxes(i, image_size=800)
            proposals = generate_gt_boxes(i, image_size=800)
            self._encode_decode(proposals, gt_boxes)


if __name__ == '__main__':
    tf.test.main()
