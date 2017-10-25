import tensorflow as tf
import numpy as np

from luminoth.models.retina import Retina


class RetinaTest(tf.test.TestCase):
    def setUp(self):
        super(RetinaTest, self).setUp()
        self._base_config = Retina.base_config
        self._model = Retina(config=self._base_config)

        self.image_size = (600, 800)
        self.image = np.random.randint(low=0, high=255, size=(1, 600, 800, 3))
        self.gt_boxes = np.array([
            [10, 10, 26, 28, 1],
            [10, 10, 20, 22, 1],
            [10, 11, 20, 21, 1],
            [19, 30, 31, 33, 1],
        ])

    def testRun(self):
        image_ph = tf.placeholder(shape=[1, None, None, 3], dtype=tf.float32)
        gt_boxes_ph = tf.placeholder(shape=[None, 5], dtype=tf.int32)
        net = self._model(image_ph, gt_boxes_ph, is_training=True)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            out_dict = sess.run(net, feed_dict={
                image_ph: self.image,
                gt_boxes_ph: self.gt_boxes
            })
            print(out_dict.keys())


if __name__ == '__main__':
    tf.test.main()
