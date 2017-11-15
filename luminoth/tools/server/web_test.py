# import numpy as np
import tensorflow as tf

from easydict import EasyDict
from luminoth.models import get_model
from luminoth.utils.config import get_base_config


class WebTest(tf.test.TestCase):
    # TODO When the image size has big dimensions like (1024, 1024, 3),
    # Travis fails during this test, probably ran out of memory. Using an build
    # environment with more memory all works fine.
    def setUp(self):
        tf.reset_default_graph()
        model_class = get_model('fasterrcnn')
        base_config = get_base_config(model_class)
        image_resize = base_config.dataset.image_preprocessing
        self.config = EasyDict({
            'image_resize_min': image_resize.min_size,
            'image_resize_max': image_resize.max_size
        })

    # # This test fails with Travis' build environment
    # def testWithoutResize(self):
    #     """
    #     Tests the FasterRCNN's predict without resize an image
    #     """
    #     # Does a prediction without resizing the image
    #     image = Image.fromarray(
    #         np.random.randint(
    #             low=0, high=255,
    #             size=(self.config.image_resize_min,
    #                   self.config.image_resize_max, 3)
    #         ).astype(np.uint8)
    #     )

    #     results = get_prediction('fasterrcnn', image)

    #     # Check that scale_factor and inference_time are corrects values
    #     self.assertEqual(results['scale_factor'], 1.0)
    #     self.assertGreaterEqual(results['inference_time'], 0)

    #     # Check that objects, labels and probs aren't None
    #     self.assertIsNotNone(results['objects'])
    #     self.assertIsNotNone(results['objects_labels'])
    #     self.assertIsNotNone(results['objects_labels_prob'])

    # This test fails with Travis' build environment
    # def testWithResize(self):
    #     """
    #     Tests the FasterRCNN's predict without resize an image
    #     """
    #     # Does a prediction resizing the image
    #     image = Image.fromarray(
    #         np.random.randint(
    #             low=0, high=255,
    #             size=(self.config.image_resize_min,
    #                   self.config.image_resize_max + 1, 3)
    #         ).astype(np.uint8)
    #     )
    #
    #     results = get_prediction('fasterrcnn', image)
    #
    #     # Check that scale_factor and inference_time are corrects values
    #     self.assertNotEqual(1.0, results['scale_factor'])
    #     self.assertGreaterEqual(results['inference_time'], 0)
    #
    #     # Check that objects, labels and probs aren't None
    #     self.assertIsNotNone(results['objects'])
    #     self.assertIsNotNone(results['objects_labels'])
    #     self.assertIsNotNone(results['objects_labels_prob'])


if __name__ == '__main__':
    tf.test.main()
