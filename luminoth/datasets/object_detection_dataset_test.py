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
                'seed': None,
            }
        })
        tf.reset_default_graph()

    def _run_augment(self, augment_config, image, bboxes):
        self.base_config['dataset']['data_augmentation'] = augment_config

        bboxes_tf = tf.placeholder(tf.int32, shape=bboxes.shape)
        image_tf = tf.placeholder(tf.int32, shape=image.shape)

        model = ObjectDetectionDataset(self.base_config)
        image_aug, bboxes_aug, applied_data_augmentation = model._augment(
            image_tf, bboxes_tf)

        with self.test_session() as sess:
            image_aug, bboxes_aug, applied_data_augmentation = sess.run(
                [image_aug, bboxes_aug, applied_data_augmentation], feed_dict={
                    bboxes_tf: bboxes,
                    image_tf: image,
                })
            return image_aug, bboxes_aug, applied_data_augmentation

    def testSortedAugmentation(self):
        """
        Tests that the augmentation is applied in order
        """
        image = np.random.randint(low=0, high=255, size=(600, 800, 3))
        bboxes = np.array([
            [10, 10, 26, 28, 1],
            [10, 10, 20, 22, 1],
            [10, 11, 20, 21, 1],
            [19, 30, 31, 33, 1],
        ])
        config = [{'flip': {'prob': 0}}, {'flip': {'prob': 1}}]

        image_aug, bboxes_aug, aug = self._run_augment(config, image, bboxes)
        self.assertEqual(aug[0], {'flip': False})
        self.assertEqual(aug[1], {'flip': True})

        config = [{'flip': {'prob': 1}}, {'flip': {'prob': 0}}]

        image_aug, bboxes_aug, aug = self._run_augment(config, image, bboxes)
        self.assertEqual(aug[0], {'flip': True})
        self.assertEqual(aug[1], {'flip': False})

    def testIdentityAugmentation(self):
        """
        Tests that to apply flip twice to an image and bboxes returns the same
        image and bboxes
        """
        image = np.random.randint(low=0, high=255, size=(600, 800, 3))
        bboxes = np.array([
            [10, 10, 26, 28, 1],
            [19, 30, 31, 33, 1],
        ])
        config = [{'flip': {'prob': 1}}, {'flip': {'prob': 1}}]

        image_aug, bboxes_aug, aug = self._run_augment(config, image, bboxes)
        self.assertEqual(aug[0], {'flip': True})
        self.assertEqual(aug[1], {'flip': True})

        self.assertAllEqual(image, image_aug)
        self.assertAllEqual(bboxes, bboxes_aug)


if __name__ == '__main__':
    tf.test.main()
