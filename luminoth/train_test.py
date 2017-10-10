import sonnet as snt
import tensorflow as tf

from easydict import EasyDict
from luminoth.train import run
from luminoth.utils.config import get_base_config


class MockFasterRCNN(snt.AbstractModule):
    """
    Mocks Faster RCNN Network
    """
    base_config = get_base_config('luminoth/models/fasterrcnn/')

    def __init__(self, config, name='mockfasterrcnn'):
        super(MockFasterRCNN, self).__init__(name=name)
        self._config = config

    def _build(self, image, gt_boxes=None, is_training=False):
        w = tf.get_variable('w', initializer=[2.5, 3.0], trainable=True)
        return {'w': w}

    def loss(self, pred_dict, return_all=False):
        return tf.reduce_sum(pred_dict['w'], 0)

    def get_trainable_vars(self):
        return snt.get_variables_in_module(self)

    def load_pretrained_weights(self):
        return tf.no_op()

    @property
    def summary(self):
        return tf.summary.scalar('dummy', 1)


class TrainTest(tf.test.TestCase):
    """
    Basic test to train module
    """
    def setUp(self):
        self.config = EasyDict({
            'model_type': 'fasterrcnn',
            'dataset_type': '',
            'config_files': [],
            'override_params': ['train.num_epochs=2'],
            'base_network': {
                'download': False
            }
        })

    def get_dataset(self, dataset_type):
        """
        Mocks luminoth.datasets.datasets.get_dataset
        """
        def dataset_class(arg2):
            def build():
                queue_dtypes = [tf.float32, tf.int32, tf.string]
                queue_names = ['image', 'bboxes', 'filename']

                queue = tf.FIFOQueue(
                    capacity=3,
                    dtypes=queue_dtypes,
                    names=queue_names,
                    name='fifo_queue'
                )
                filename = tf.cast('filename_test', tf.string)
                filename = tf.train.limit_epochs([filename], num_epochs=2)

                data = {
                    'image': tf.random_uniform([600, 800, 3], maxval=255),
                    'bboxes': tf.constant([[0, 0, 30, 30, 0]]),
                    'filename': filename
                }
                enqueue_ops = [queue.enqueue(data)] * 2
                tf.train.add_queue_runner(
                    tf.train.QueueRunner(queue, enqueue_ops))

                return queue.dequeue()
            return build
        return dataset_class

    def get_model(self, model_type):
        """
        Mocks from luminoth.models.get_model
        """
        return MockFasterRCNN

    def testTrain(self):
        config = self.config

        # This should not fail
        run(config.config_files, config.override_params,
            get_dataset_fn=self.get_dataset, get_model_fn=self.get_model)


if __name__ == '__main__':
    tf.test.main()
