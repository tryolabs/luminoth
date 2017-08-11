import numpy as np
import tensorflow as tf

from luminoth.utils.bbox_transform import encode, decode, clip_boxes
from luminoth.utils.test.gt_boxes import generate_gt_boxes


class BBoxTransformTest(tf.test.TestCase):
    def testEncodeDecode(self):
        # 1 vs 1 already equal encode and decode.
        proposal = generate_gt_boxes(1, image_size=100)
        gt_boxes = proposal

        deltas = encode(proposal, gt_boxes)
        decoded_gt_boxes = decode(proposal, deltas)

        self.assertAllEqual(deltas, np.zeros((1, 4)))
        self.assertAllClose(gt_boxes, decoded_gt_boxes)

        # 3 vs 3 already equal encode and decode
        proposal = generate_gt_boxes(3, image_size=100)
        gt_boxes = proposal

        deltas = encode(proposal, gt_boxes)
        decoded_gt_boxes = decode(proposal, deltas)

        self.assertAllEqual(deltas, np.zeros((3, 4)))
        self.assertAllClose(gt_boxes, decoded_gt_boxes)

        # 3 vs 4 different encode and decode
        proposal = generate_gt_boxes(3, image_size=100)
        gt_boxes = generate_gt_boxes(3, image_size=100)

        deltas = encode(proposal, gt_boxes)
        decoded_gt_boxes = decode(proposal, deltas)

        self.assertAllEqual(deltas.shape, (3, 4))
        self.assertAllClose(gt_boxes, decoded_gt_boxes)

    def testClipBboxes(self):
        image_shape = (100, 100)  # height, width
        boxes = np.array([
            [-1, 10, 20, 20],  # x_min is left of the image.
            [10, -1, 20, 20],  # y_min is above the image.
            [10, 10, 100, 20],  # x_max is right of the image.
            [10, 10, 20, 100],  # y_max is below the image.
            [10, 10, 20, 20],  # everything is in place
            [100, 100, 101, 101],  # complete box is outside the image.
        ])
        clipped_bboxes = clip_boxes(boxes, image_shape)
        self.assertAllEqual(clipped_bboxes, [
            [0, 10, 20, 20],
            [10, 0, 20, 20],
            [10, 10, 99, 20],
            [10, 10, 20, 99],
            [10, 10, 20, 20],
            [99, 99, 99, 99],
        ])


if __name__ == '__main__':
    tf.test.main()
