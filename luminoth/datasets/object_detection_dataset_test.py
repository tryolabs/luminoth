import tensorflow as tf
import numpy as np

from easydict import EasyDict

from luminoth.datasets.object_detection_dataset import ObjectDetectionDataset


class ObjectDetectionDatasetTest(tf.test.TestCase):
    def setUp(self):
        self.base_config = EasyDict({
            'dataset': {
                'dir': '',
                'split': 'train',
                'image_preprocessing': {
                    'min_size': 600,
                    'max_size': 1024,
                },
                'data_augmentation': {},
            },
            'train': {
                'num_epochs': 1,
                'batch_size': 1,
                'random_shuffle': False,
            }
        })

    def _run_augment(self, augment_config):
        self.base_config['dataset']['data_augmentation'] = augment_config

        bboxes = tf.placeholder(tf.int32, shape=self.bboxes.shape)
        image = tf.placeholder(tf.int32, shape=self.image.shape)

        model = ObjectDetectionDataset(self.base_config)
        image, bboxes, applied_data_augmentation = model._augment(
            image, bboxes)

        with self.test_session() as sess:
            image, bboxes, applied_data_augmentation = sess.run(
                [image, bboxes, applied_data_augmentation], feed_dict={
                    bboxes: self.bboxes,
                    image: self.image,
                })
            return image, bboxes, applied_data_augmentation

    def testSortedAugmentation(self):
        """
        Tests that the augmentation is applied in order
        """
        self.image = np.random.randint(low=0, high=255, size=(600, 800, 3))
        self.bboxes = np.array([
            [10, 10, 26, 28, 1],
            [10, 10, 20, 22, 1],
            [10, 11, 20, 21, 1],
            [19, 30, 31, 33, 1],
        ])
        config = [{'flip': {'prob': 0}}, {'flip': {'prob': 1}}]

        image, bboxes, aug = self._run_augment(config)
        self.assertEqual(aug[0], {'flip': False})
        self.assertEqual(aug[1], {'flip': True})

        config = [{'flip': {'prob': 1}}, {'flip': {'prob': 0}}]

        image, bboxes, aug = self._run_augment(config)
        self.assertEqual(aug[0], {'flip': True})
        self.assertEqual(aug[1], {'flip': False})

    def testIdentityAugmentation(self):
        """
        Tests that to apply flip twice to an image and bboxes returns the same
        image and bboxes
        """
        self.image = np.random.randint(low=0, high=255, size=(600, 800, 3))
        self.bboxes = np.array([
            [10, 10, 26, 28, 1],
            [19, 30, 31, 33, 1],
        ])
        config = [{'flip': {'prob': 1}}, {'flip': {'prob': 1}}]

        image, bboxes, aug = self._run_augment(config)
        self.assertEqual(aug[0], {'flip': True})
        self.assertEqual(aug[1], {'flip': True})

        self.assertAllEqual(self.image, image)
        self.assertAllEqual(self.bboxes, bboxes)


if __name__ == '__main__':
    tf.test.main()
