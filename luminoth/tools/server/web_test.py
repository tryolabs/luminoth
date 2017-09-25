import numpy as np
import tensorflow as tf

from PIL import Image
from luminoth.models import get_model
from luminoth.tools.server.web import get_prediction


class WebTest(tf.test.TestCase):

    def testFasterRCNN(self):
        """
        Tests the FasterRCNN's predict
        """
        model_class = get_model('fasterrcnn')
        image_resize = model_class.base_config.dataset.image_preprocessing
        image_resize_min = image_resize.min_size
        image_resize_max = image_resize.max_size

        # Does a prediction without resizing the image
        image = Image.fromarray(
            np.random.randint(
                low=0, high=255,
                size=(image_resize_min + 100, image_resize_max - 100, 3)
            ).astype(np.uint8)
        )
        results = get_prediction('fasterrcnn', image)

        # Check that scale_factor and inference_time are corrects values
        self.assertEqual(results['scale_factor'], 1.0)
        self.assertGreaterEqual(results['inference_time'], 0)

        # Check that objects, labels and probs aren't None
        self.assertNotEqual(results['objects'], None)
        self.assertNotEqual(results['objects_labels'], None)
        self.assertNotEqual(results['objects_labels_prob'], None)

        # Does a prediction resizing the image
        image = Image.fromarray(
            np.random.randint(
                low=0, high=255,
                size=(image_resize_max + 100, image_resize_max + 100, 3)
            ).astype(np.uint8)
        )
        results = get_prediction('fasterrcnn', image)

        # Check that scale_factor and inference_time are corrects values
        self.assertGreaterEqual(1.0, results['scale_factor'])
        self.assertGreaterEqual(results['inference_time'], 0)

        # Check that objects, labels and probs aren't None
        self.assertNotEqual(results['objects'], None)
        self.assertNotEqual(results['objects_labels'], None)
        self.assertNotEqual(results['objects_labels_prob'], None)


if __name__ == '__main__':
    tf.test.main()
