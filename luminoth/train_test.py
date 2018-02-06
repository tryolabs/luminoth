import sonnet as snt
import tempfile
import tensorflow as tf

from easydict import EasyDict
from luminoth.train import run
from luminoth.models import get_model
from luminoth.utils.config import (
    get_model_config, load_config_files, get_base_config
)


class MockFasterRCNN(snt.AbstractModule):
    """
    Mocks Faster RCNN Network
    """
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
        return tf.summary.scalar('dummy', 1, collections=['rcnn'])


class TrainTest(tf.test.TestCase):
    """
    Basic test to train module
    """
    def setUp(self):
        self.total_epochs = 2
        self.config = EasyDict({
            'model_type': 'fasterrcnn',
            'dataset_type': '',
            'config_files': (),
            'override_params': [],
            'base_network': {
                'download': False
            }
        })
        tf.reset_default_graph()

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

    def get_config(self, model_type, override_params=None):
        custom_config = load_config_files(self.config.config_files)
        model_class = get_model('fasterrcnn')
        model_base_config = get_base_config(model_class)
        config = get_model_config(
            model_base_config, custom_config, override_params
        )

        config.model.type = model_type

        return config

    def testTrain(self):
        model_type = 'mockfasterrcnn'

        override_params = [
            'train.num_epochs={}'.format(self.total_epochs),
            'train.job_dir=',
        ]

        config = self.get_config(model_type, override_params=override_params)

        # This should not fail
        run(
            config, get_dataset_fn=self.get_dataset,
            get_model_fn=self.get_model
        )

    def testTrainSave(self):
        model_type = 'mockfasterrcnn'

        # Save checkpoints to a temp directory.
        tmp_job_dir = tempfile.mkdtemp()
        override_params = [
            'train.num_epochs={}'.format(self.total_epochs),
            'train.job_dir={}'.format(tmp_job_dir),
            'train.run_name=test_runname',
        ]

        config = self.get_config(model_type, override_params=override_params)

        step = run(
            config,
            get_dataset_fn=self.get_dataset, get_model_fn=self.get_model
        )
        self.assertEqual(step, 2)

        # We have to reset the graph to avoid having duplicate names.
        tf.reset_default_graph()
        step = run(
            config,
            get_dataset_fn=self.get_dataset, get_model_fn=self.get_model
        )

        # This is because of a MonitoredTrainingSession "bug".
        # When ending training it saves a checkpoint as the next step.
        # That causes that we are one step ahead when loading it.
        self.assertEqual(step, 5)


if __name__ == '__main__':
    tf.test.main()
