import numpy as np
import tensorflow as tf

from luminoth.utils.bbox_transform import encode, decode
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


if __name__ == '__main__':
    tf.test.main()
