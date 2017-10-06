import tensorflow as tf

from easydict import EasyDict
from luminoth.train import run


class TrainTest(tf.test.TestCase):
    def setUp(self):
        self.config = EasyDict({
            'model_type': 'fasterrcnn',
            'dataset_type': '',
            'config_file': None,
            'override_params': ['train.num_epochs=2'],
            'run_name': 'vgg',
            'save_summaries_secs': None,
            'base_network': {
                'download': False
            }
        })

    def get_dataset(self, dataset_type):
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

    def testTrain(self):
        config = self.config

        # This should not fail
        run(config.model_type, config.dataset_type, config.config_file,
            config.override_params, run_name=config.run_name,
            save_summaries_secs=config.save_summaries_secs,
            get_dataset=self.get_dataset)


if __name__ == '__main__':
    tf.test.main()
