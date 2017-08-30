import numpy as np
import tensorflow as tf

from easydict import EasyDict

from luminoth.utils.image import (
    resize_image, flip_image, random_patch
)
from luminoth.utils.test.gt_boxes import generate_gt_boxes


class ImageTest(tf.test.TestCase):
    def setUp(self):
        self._random_patch_config = EasyDict({
            'min_height': 400,
            'min_width': 400,
        })

    def _gen_image(self, *shape):
        return np.random.rand(*shape)

    def _get_image_with_boxes(self, image_size, total_boxes):
        image = self._gen_image(*image_size)
        bboxes = generate_gt_boxes(
            total_boxes, image_size[:2],
        )
        return image, bboxes

    def _resize_image(self, image_array, boxes_array=None, min_size=None,
                      max_size=None):
        image = tf.placeholder(tf.float32, image_array.shape)
        feed_dict = {
            image: image_array,
        }
        if boxes_array is not None:
            boxes = tf.placeholder(tf.float32, boxes_array.shape)
            feed_dict[boxes] = boxes_array
        else:
            boxes = None
        resized = resize_image(
            image, bboxes=boxes, min_size=min_size, max_size=max_size
        )
        with self.test_session() as sess:
            resized_dict = sess.run(resized, feed_dict=feed_dict)
            return (
                resized_dict['image'],
                resized_dict.get('bboxes'),
                resized_dict.get('scale_factor'),
            )

    def _flip_image(self, image_array, boxes_array=None, left_right=False,
                    up_down=False, bboxes_dtype=tf.int32):
        image = tf.placeholder(tf.float32, image_array.shape)
        feed_dict = {
            image: image_array,
        }
        config = EasyDict({
            'left_right': left_right,
            'up_down': up_down,
        })
        if boxes_array is not None:
            boxes = tf.placeholder(bboxes_dtype, boxes_array.shape)
            feed_dict[boxes] = boxes_array
        else:
            boxes = None
        flipped = flip_image(
            image, config, bboxes=boxes,
        )
        with self.test_session() as sess:
            flipped_dict = sess.run(flipped, feed_dict=feed_dict)
            return flipped_dict['image'], flipped_dict.get('bboxes')

    def _random_patch(self, image, config, bboxes=None):
        with self.test_session() as sess:
            # passing bboxes=None throws an error.
            patch = random_patch(image, config, bboxes=bboxes, debug=True)
            return_dict = sess.run(patch)
            ret_bboxes = return_dict.get('bboxes')
            return return_dict['image'], ret_bboxes

    def testResizeOnlyImage(self):
        # No min or max size, it doesn't change the image.
        resized_image, _, scale = self._resize_image(
            self._gen_image(100, 1024, 3)
        )
        self.assertAllEqual(
            resized_image.shape,
            (100, 1024, 3)
        )
        self.assertAlmostEqual(scale, 1.)

        # min and max sizes smaller and larger that original image size.
        resized_image, _, scal = self._resize_image(
            self._gen_image(100, 1024, 3), min_size=0, max_size=2000
        )
        self.assertAllEqual(
            resized_image.shape,
            (100, 1024, 3)
        )
        self.assertAlmostEqual(scale, 1.)

        # max_size only, slightly smaller than origin image size.
        resized_image, _, scale = self._resize_image(
            self._gen_image(100, 1024, 3), max_size=1000
        )
        self.assertAllEqual(
            resized_image.shape,
            (97, 1000, 3)
        )
        self.assertAlmostEqual(int(scale * 100), 97)

        # min_size only, slightly bigger than origin image size.
        resized_image, _, scale = self._resize_image(
            self._gen_image(100, 1024, 3), min_size=120
        )
        self.assertAllEqual(
            resized_image.shape,
            (120, 1228, 3)
        )
        self.assertAlmostEqual(int(scale * 100), 120)

        # max_size only, half image size
        resized_image, _, scale = self._resize_image(
            self._gen_image(100, 1024, 3), max_size=512
        )
        self.assertAllEqual(
            resized_image.shape,
            (50, 512, 3)
        )
        self.assertAlmostEqual(scale, 0.5)

        # min_size only, double image size
        resized_image, _, scale = self._resize_image(
            self._gen_image(100, 1024, 3), min_size=200
        )
        self.assertAllEqual(
            resized_image.shape,
            (200, 2048, 3)
        )
        self.assertAlmostEqual(scale, 2.)

        # min and max invariant changes, both changes negate themselves.
        change = 1.1
        resized_image, _, scale = self._resize_image(
            self._gen_image(100, 200, 3), min_size=int(100 * change),
            max_size=round(200 / change)
        )
        self.assertAllEqual(
            resized_image.shape,
            (100, 200, 3)
        )
        self.assertAlmostEqual(int(scale), 1)

        # min and max invariant changes, both changes negate themselves.
        change = 1.6  # Not all values work because of rounding issues.
        resized_image, _, scale = self._resize_image(
            self._gen_image(100, 200, 3), min_size=int(100 * change),
            max_size=round(200 / change)
        )
        self.assertAllEqual(
            resized_image.shape,
            (100, 200, 3)  # No change
        )
        self.assertAlmostEqual(int(scale), 1)

        resized_image, _, scale = self._resize_image(
            self._gen_image(100, 200, 3), min_size=600,
            max_size=1000
        )
        self.assertAllEqual(
            resized_image.shape,
            (100 * 6, 200 * 6, 3)  # We end up bigger than maximum.
        )
        self.assertAlmostEqual(scale, 6.)

        resized_image, _, scale = self._resize_image(
            self._gen_image(2000, 600, 3), min_size=600,
            max_size=1000
        )
        self.assertAllEqual(
            resized_image.shape,
            (1000, 300, 3)  # We end up smaller than minimum image.
        )
        self.assertAlmostEqual(scale, 0.5)

    def testResizeImageBoxes(self):
        image = self._gen_image(100, 100, 3)
        # Half size, corner box
        boxes = np.array([[0, 0, 10, 10, -1]])
        _, resized_boxes, scale = self._resize_image(
            image, boxes, max_size=50
        )
        self.assertAllEqual(resized_boxes, [[0, 0, 5, 5, -1]])

        # Half size, middle box
        boxes = np.array([[10, 10, 90, 90, -1]])
        _, resized_boxes, scale = self._resize_image(
            image, boxes, max_size=50
        )
        self.assertAllEqual(resized_boxes, [[5, 5, 45, 45, -1]])

        # Quarter size, maximum box
        boxes = np.array([[0, 0, 99, 99, -1]])
        _, resized_boxes, scale = self._resize_image(
            image, boxes, max_size=25
        )
        self.assertAllEqual(resized_boxes, [[0, 0, 24, 24, -1]])

    def testFlipOnlyImage(self):
        # No changes to image or boxes when no flip is specified.
        image = self._gen_image(100, 100, 3)
        not_flipped_image, _ = self._flip_image(image)
        self.assertAllClose(not_flipped_image, image)

        # Test flipped image that original first column is equal to the last
        # column of the flipped one.
        image = self._gen_image(100, 100, 3)
        flipped_image, _ = self._flip_image(image, left_right=True)
        self.assertAllClose(flipped_image[:, 0], image[:, -1])
        self.assertAllClose(flipped_image[:, 1], image[:, -2])

    def testFlipImageBoxes(self):
        image, boxes = self._get_image_with_boxes((500, 500, 3), 10)
        not_flipped_image, not_flipped_boxes = self._flip_image(image, boxes)
        self.assertAllClose(not_flipped_image, image)
        self.assertAllClose(not_flipped_boxes, boxes)

        # Test box same size as image, left_right flip should not change.
        image = self._gen_image(100, 100, 3)
        boxes = np.array([[0, 0, 99, 99, -1]])
        flipped_image, flipped_boxes = self._flip_image(
            image, boxes, left_right=True
        )
        self.assertAllClose(boxes, flipped_boxes)
        # Check that sum of columns is not modified, just the order.
        self.assertAllClose(image.sum(axis=1), flipped_image.sum(axis=1))

        # Test box same size as image, up_down flip, should not change.
        image = self._gen_image(100, 100, 3)
        boxes = np.array([[0, 0, 99, 99, -1]])
        flipped_image, flipped_boxes = self._flip_image(
            image, boxes, up_down=True
        )
        self.assertAllClose(boxes, flipped_boxes)
        # Check that sum of columns is not modified, just the order.
        self.assertAllClose(image.sum(axis=0), flipped_image.sum(axis=0))

        # Test box same size as image, up_down flip, should not change.
        image = self._gen_image(100, 100, 3)
        boxes = np.array([[0, 0, 99, 99, -1]])
        flipped_image, flipped_boxes = self._flip_image(
            image, boxes, up_down=True, left_right=True
        )
        # Check that sum of columns is not modified, just the order.
        self.assertAllClose(boxes, flipped_boxes)

        # Test box same size as image, up_down flip, should not change.
        image = self._gen_image(100, 100, 3)
        boxes = np.array([[0, 0, 10, 10, -1]])
        flipped_image, flipped_boxes = self._flip_image(
            image, boxes, up_down=True, left_right=True
        )
        # Check that sum of columns is not modified, just the order.
        self.assertAllClose(
            np.array([[89.,  89.,  99.,  99.,  -1.]]), flipped_boxes
        )

    def testFlipBboxesDiffDtype(self):
        image = self._gen_image(100, 100, 3)
        boxes = np.array([[25, 14, 63, 41, -1]])
        _, flipped_boxes_float = self._flip_image(
            image, boxes, up_down=True, left_right=True,
            bboxes_dtype=tf.float32
        )

        _, flipped_boxes_int = self._flip_image(
            image, boxes, up_down=True, left_right=True,
            bboxes_dtype=tf.int32
        )

        # Check that using different types is exactly the same.
        self.assertAllClose(
            flipped_boxes_float, flipped_boxes_int
        )

    def testRandomPatchImageBboxes(self):
        """Tests the integrity of the return values of random_patch

        When bboxes is not None.
        """
        im_shape = (800, 600, 3)
        total_boxes = 20
        # We don't care about the label
        label = 3
        # First test case, we use randomly generated image and bboxes.
        image, bboxes = self._get_image_with_boxes(im_shape, total_boxes)
        # Add a label to each bbox.
        bboxes_w_label = tf.concat(
            [
                bboxes,
                tf.fill((bboxes.shape[0], 1), label)
            ],
            axis=1
        )
        config = self._random_patch_config
        ret_image, ret_bboxes = self._random_patch(
            image, config, bboxes_w_label
        )
        # Assertions
        self.assertLessEqual(ret_bboxes.shape[0], total_boxes)
        self.assertTrue(np.all(ret_bboxes >= 0))
        self.assertTrue(np.all(ret_bboxes[:, :4] <= ret_image.shape[1]))
        self.assertTrue(np.all(ret_image.shape <= im_shape))

    def testRandomPatchOnlyImage(self):
        """Tests the integrity of the return values of random_patch

        When bboxes is None.
        """
        im_shape = (600, 800, 3)
        image = self._gen_image(*im_shape)
        config = self._random_patch_config
        ret_image, ret_bboxes = self._random_patch(image, config)
        # Assertions
        self.assertTrue(np.all(ret_image.shape <= im_shape))
        self.assertIs(ret_bboxes, None)


if __name__ == '__main__':
    tf.test.main()
