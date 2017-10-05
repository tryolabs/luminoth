import numpy as np
import tensorflow as tf

from easydict import EasyDict

from luminoth.utils.image import (
    resize_image, flip_image, random_patch, random_resize, random_distortion,
    patch_image
)
from luminoth.utils.test.gt_boxes import generate_gt_boxes


class ImageTest(tf.test.TestCase):
    def setUp(self):
        self._random_resize_config = EasyDict({
            'min_size': 400,
            'max_size': 980,
        })
        self._random_distort_config = EasyDict({
            'brightness': {
                'max_delta': 0.3,
            },
            'contrast': {
                'lower': 0.4,
                'upper': 0.8,
            },
            'hue': {
                'max_delta': 0.2,
            },
            'saturation': {
                'lower': 0.5,
                'upper': 1.5,
            }
        })
        self._random_patch_config = EasyDict({
            'min_height': 600,
            'min_width': 600,
        })
        tf.reset_default_graph()

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
        if boxes_array is not None:
            boxes = tf.placeholder(bboxes_dtype, boxes_array.shape)
            feed_dict[boxes] = boxes_array
        else:
            boxes = None
        flipped = flip_image(
            image, bboxes=boxes, left_right=left_right, up_down=up_down
        )
        with self.test_session() as sess:
            flipped_dict = sess.run(flipped, feed_dict=feed_dict)
            return flipped_dict['image'], flipped_dict.get('bboxes')

    def _random_patch(self, image, config, bboxes=None):
        with self.test_session() as sess:
            image = tf.cast(image, tf.float32)
            if bboxes is not None:
                bboxes = tf.cast(bboxes, tf.int32)
            patch = random_patch(image, bboxes=bboxes, seed=0, **config)
            return_dict = sess.run(patch)
            ret_bboxes = return_dict.get('bboxes')
            return return_dict['image'], ret_bboxes

    def _random_resize(self, image, config, bboxes=None):
        config = self._random_resize_config
        with self.test_session() as sess:
            resize = random_resize(image, bboxes=bboxes, seed=0, **config)
            return_dict = sess.run(resize)
            ret_bboxes = return_dict.get('bboxes')
            return return_dict['image'], ret_bboxes

    def _random_distort(self, image, config, bboxes=None):
        with self.test_session() as sess:
            distort = random_distortion(
                image, bboxes=bboxes, seed=0, **config
            )
            return_dict = sess.run(distort)
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

    def testPatchImageUpdateCondition(self):
        """Tests we're not patching if we would lose all gt_boxes.
        """
        im_shape = (600, 800, 3)
        label = 3
        image_ph = tf.placeholder(shape=(None, None, 3), dtype=tf.float32)
        bboxes_ph = tf.placeholder(shape=(None, 5), dtype=tf.int32)

        with self.test_session() as sess:
            # Generate image and bboxes.
            image = self._gen_image(*im_shape)
            bboxes = [(0, 0, 40, 40, label), (430, 200, 480, 250, label)]
            # Get patch_image so that the proposed patch has no gt_box center
            # in it.
            patch = patch_image(
                image_ph, bboxes_ph,
                offset_height=45, offset_width=45,
                target_height=100, target_width=200
            )
            feed_dict = {
                image_ph: image,
                bboxes_ph: bboxes
            }
            # Run patch in a test session.
            ret_dict = sess.run(patch, feed_dict)

            ret_image = ret_dict['image']
            ret_bboxes = ret_dict.get('bboxes')
        # Assertions
        self.assertAllClose(ret_image, image)
        self.assertAllClose(ret_bboxes, bboxes)

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
        total_boxes = 5
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
        self.assertGreater(ret_bboxes.shape[0], 0)
        self.assertTrue(np.all(ret_bboxes >= 0))
        self.assertTrue(np.all(
            ret_bboxes[:, 0] <= ret_image.shape[1]
        ))
        self.assertTrue(np.all(
            ret_bboxes[:, 1] <= ret_image.shape[0]
        ))
        self.assertTrue(np.all(
            ret_bboxes[:, 2] <= ret_image.shape[1]
        ))
        self.assertTrue(np.all(
            ret_bboxes[:, 3] <= ret_image.shape[0]
        ))
        self.assertTrue(np.all(ret_image.shape <= im_shape))

    def testRandomPatchLargerThanImage(self):
        """Tests random_patch normalizes the minimum sizes.
        """
        im_shape = (600, 800, 3)
        total_boxes = 5
        config = EasyDict({
            'min_height': 900,
            'min_width': 900
        })
        label = 3
        image, bboxes = self._get_image_with_boxes(im_shape, total_boxes)
        # Add a label to each bbox.
        bboxes_w_label = tf.concat(
            [
                bboxes,
                tf.fill((bboxes.shape[0], 1), label)
            ],
            axis=1
        )
        ret_image, ret_bboxes = self._random_patch(
            image, config, bboxes_w_label
        )
        # Assertions
        self.assertLessEqual(ret_bboxes.shape[0], total_boxes)
        self.assertGreater(ret_bboxes.shape[0], 0)
        self.assertTrue(np.all(ret_bboxes >= 0))
        self.assertTrue(np.all(
            ret_bboxes[:, 0] <= ret_image.shape[1]
        ))
        self.assertTrue(np.all(
            ret_bboxes[:, 1] <= ret_image.shape[0]
        ))
        self.assertTrue(np.all(
            ret_bboxes[:, 2] <= ret_image.shape[1]
        ))
        self.assertTrue(np.all(
            ret_bboxes[:, 3] <= ret_image.shape[0]
        ))
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
        # We ran return_dict.get('bboxes') on the dict returned by
        # random_patch. That's why we should get a None in this case.
        self.assertIs(ret_bboxes, None)

    def testRandomResizeImageBboxes(self):
        """Tests the integrity of the return values of random_resize

        This tests the case when bboxes is not None.
        """
        im_shape = (600, 800, 3)
        config = self._random_resize_config
        total_boxes = 5
        label = 3

        image, bboxes = self._get_image_with_boxes(im_shape, total_boxes)
        # Add a label to each bbox.
        bboxes_w_label = tf.concat(
            [
                bboxes,
                tf.fill((bboxes.shape[0], 1), label)
            ],
            axis=1
        )
        ret_image, ret_bboxes = self._random_resize(
            image, config, bboxes_w_label
        )
        # Assertions
        self.assertEqual(ret_bboxes.shape[0], total_boxes)
        self.assertTrue(np.all(
            np.asarray(ret_image.shape[:2]) >= config.min_size
        ))
        self.assertTrue(np.all(
            np.asarray(ret_image.shape[:2]) <= config.max_size
        ))

    def testRandomResizeOnlyImage(self):
        """Tests the integrity of the return values of random_resize

        This tests the case when bboxes is None.
        """
        im_shape = (600, 800, 3)
        image = self._gen_image(*im_shape)
        config = self._random_resize_config
        ret_image, ret_bboxes = self._random_resize(image, config)
        # Assertions
        self.assertEqual(ret_bboxes, None)
        self.assertTrue(np.all(
            np.asarray(ret_image.shape[:2]) >= config.min_size
        ))
        self.assertTrue(np.all(
            np.asarray(ret_image.shape[:2]) <= config.max_size
        ))

    def testRandomDistort(self):
        """Tests the integrity of the return values of random_distortion.
        """
        im_shape = (600, 900, 3)
        config = self._random_distort_config
        total_boxes = 5
        label = 3

        image, bboxes = self._get_image_with_boxes(im_shape, total_boxes)
        # Add a label to each bbox.
        bboxes_w_label = tf.concat(
            [
                bboxes,
                tf.fill((bboxes.shape[0], 1), label)
            ],
            axis=1
        )

        ret_image, ret_bboxes = self._random_distort(
            image, config, bboxes_w_label
        )
        # Assertions
        self.assertEqual(im_shape, ret_image.shape)
        self.assertAllEqual(
            bboxes, ret_bboxes[:, :4]
        )

    def testSmallRandomDistort(self):
        """Tests random_distort with small-change arguments.

        We pass parameters to random_distort that make it so that it should
        change the image relatively little, and then check that in fact it
        changed relatively little.
        """
        total_boxes = 3
        im_shape = (600, 900, 3)
        config = EasyDict({
            'brightness': {
                'max_delta': 0.00001,
            },
            'hue': {
                'max_delta': 0.00001,
            },
            'saturation': {
                'lower': 0.99999,
                'upper': 1.00001,
            },
            'contrast': {
                'lower': 0.99999,
                'upper': 1.00001
            }
        })
        label = 3
        image, bboxes = self._get_image_with_boxes(im_shape, total_boxes)
        # Add a label to each bbox.
        bboxes_w_label = tf.concat(
            [
                bboxes,
                tf.fill((bboxes.shape[0], 1), label)
            ],
            axis=1
        )
        ret_image, ret_bboxes = self._random_distort(
            image, config, bboxes_w_label
        )
        # Assertions
        large_number = 0.1
        self.assertAllClose(image, ret_image, rtol=0.05, atol=large_number)


if __name__ == '__main__':
    tf.test.main()
