import numpy as np
import tensorflow as tf

from easydict import EasyDict
from luminoth.models.fasterrcnn.rcnn import RCNN


class MockBaseNetwork():

    def _build_tail(self, features, **kargs):
        return features


class RCNNTest(tf.test.TestCase):
    def setUp(self):
        tf.reset_default_graph()

        self._num_classes = 5
        self._num_proposals = 256
        self._total_num_gt = 128
        self._image_shape = (600, 800)
        # The score we'll give to the true labels when testing for perfect
        # score generation.
        self._high_score = 100

        self._equality_delta = 1e-03

        self._config = EasyDict({
            'enabled': True,
            'layer_sizes': [4096, 4096],
            'dropout_keep_prob': 1.0,
            'activation_function': 'relu6',
            'use_mean': False,
            'rcnn_initializer': {
                'type': 'variance_scaling_initializer',
                'factor': 1.0,
                'uniform': True,
                'mode': 'FAN_AVG',
            },
            'bbox_initializer': {
                'type': 'variance_scaling_initializer',
                'factor': 1.0,
                'uniform': True,
                'mode': 'FAN_AVG',
            },
            'cls_initializer': {
                'type': 'variance_scaling_initializer',
                'factor': 1.0,
                'uniform': True,
                'mode': 'FAN_AVG',
            },
            'l2_regularization_scale': 0.0005,
            'l1_sigma': 3.0,
            'roi': {
                'pooling_mode': 'crop',
                'pooled_width': 7,
                'pooled_height': 7,
                'padding': 'VALID',
            },
            'proposals': {
                'class_max_detections': 100,
                'class_nms_threshold': 0.6,
                'total_max_detections': 300,
                'min_prob_threshold': 0.0,
            },
            'target': {
                'foreground_fraction': 0.25,
                'minibatch_size': 64,
                'foreground_threshold': 0.5,
                'background_threshold_high': 0.5,
                'background_threshold_low': 0.1,
            },

        })

        self._base_network = MockBaseNetwork()
        self._shared_model = RCNN(self._num_classes, self._config)

        # Declare placeholders
        # We use the '_ph' suffix for placeholders.
        self._pretrained_feature_map_shape = (
            self._num_proposals,
            self._config.roi.pooled_width,
            self._config.roi.pooled_height,
            4
        )
        self._pretrained_feature_map_ph = tf.placeholder(
            tf.float32, shape=self._pretrained_feature_map_shape
        )

        self._proposals_shape = (self._num_proposals, 4)
        self._proposals_ph = tf.placeholder(
            tf.float32, shape=self._proposals_shape
        )

        self._image_shape_shape = (2,)
        self._image_shape_ph = tf.placeholder(
            tf.float32, shape=self._image_shape_shape
        )

        self._gt_boxes_shape = (self._total_num_gt, 5)
        self._gt_boxes_ph = tf.placeholder(
            tf.float32, shape=self._gt_boxes_shape
        )

    def _run_net_with_feed_dict(self, net, feed_dict):
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            return sess.run(net, feed_dict=feed_dict)

    def _check_returning_shapes(self, prediction_dict, training=False):
        """Asserts a prediction_dict has the right shapes.

        This includes testing that:
            - objects, objects_labels and objects_labels_prob have the same
                shape in the first dimension. (i.e. the same number of
                objects).
            - objects has shape (_, 4). objects_labels and objects_labels_prob
                have shape (_,).
            - cls_score and cls_prob have shape (num_proposals,
                num_classes + 1).
            - bbox_offsets has shape (num_proposals, num_classes * 4).

        And, if training:
            - cls_target has shape (num_proposals,).
            - bbox_offsets_target has shape (num_proposals, 4).
        """

        objects_shape = prediction_dict['objects'].shape
        objects_labels_shape = prediction_dict['labels'].shape
        objects_labels_prob_shape = prediction_dict['probs'] \
            .shape

        cls_score_shape = prediction_dict['rcnn']['cls_score'].shape
        cls_prob_shape = prediction_dict['rcnn']['cls_prob'].shape

        bbox_offsets_shape = prediction_dict['rcnn']['bbox_offsets'].shape

        # We choose cls_score as the 'standard' num_proposals which we will
        # compare to the other shapes that should include num_proposals. We
        # could have chosen a different one.
        num_proposals = cls_score_shape[0]

        self.assertEqual(objects_shape[0], objects_labels_shape[0])
        self.assertEqual(objects_shape[0], objects_labels_prob_shape[0])

        self.assertEqual(objects_shape[1], 4)
        self.assertEqual(len(objects_labels_shape), 1)
        self.assertEqual(len(objects_labels_prob_shape), 1)

        self.assertEqual(cls_score_shape, cls_prob_shape)
        self.assertEqual(
            cls_prob_shape,
            (num_proposals, self._num_classes + 1)
        )

        self.assertEqual(
            bbox_offsets_shape,
            (num_proposals, self._num_classes * 4)
        )

        if training:
            cls_target_shape = prediction_dict['target']['cls'].shape
            self.assertEqual(cls_target_shape, (num_proposals,))

            bbox_offsets_trgt_shape = (
                prediction_dict['target']['bbox_offsets'].shape
            )
            self.assertEqual(
                bbox_offsets_trgt_shape,
                (num_proposals, 4)
            )

    def testReturningShapes(self):
        """Tests we're returning consistent shapes.

        We test both the case where we're training and the case where we are
        not.
        """

        # Prediction session (not training)
        rcnn_net_not_training = self._shared_model(
            self._pretrained_feature_map_ph, self._proposals_ph,
            self._image_shape_ph, self._base_network
        )

        prediction_dict_not_training = self._run_net_with_feed_dict(
            rcnn_net_not_training,
            feed_dict={
                self._pretrained_feature_map_ph: np.random.rand(
                    *self._pretrained_feature_map_shape
                ),
                self._proposals_ph: np.random.randint(
                    low=0,
                    high=np.amin(self._image_shape),
                    size=self._proposals_shape,
                ),
                self._image_shape_ph: self._image_shape,
            }
        )
        # Training session
        rcnn_net_training = self._shared_model(
            self._pretrained_feature_map_ph, self._proposals_ph,
            self._image_shape_ph, self._base_network, self._gt_boxes_ph
        )
        prediction_dict_training = self._run_net_with_feed_dict(
            rcnn_net_training,
            feed_dict={
                self._pretrained_feature_map_ph: np.random.rand(
                    *self._pretrained_feature_map_shape
                ),
                self._proposals_ph: np.random.randint(
                    low=0,
                    high=np.amin(self._image_shape),
                    size=self._proposals_shape,
                ),
                self._image_shape_ph: self._image_shape,
                self._gt_boxes_ph: np.random.randint(
                    low=0,
                    high=np.amin(self._image_shape),
                    size=self._gt_boxes_shape,
                ),
            }
        )
        # Assertions
        self._check_returning_shapes(
            prediction_dict_not_training
        )
        self._check_returning_shapes(
            prediction_dict_training, training=True
        )

    def testMinibatchBehaviour(self):
        """Tests we're using minibatch_size correctly when testing.
        """

        rcnn_net = self._shared_model(
            self._pretrained_feature_map_ph, self._proposals_ph,
            self._image_shape_ph, self._base_network, self._gt_boxes_ph
        )

        prediction_dict = self._run_net_with_feed_dict(
            rcnn_net,
            feed_dict={
                self._pretrained_feature_map_ph: np.random.rand(
                    *self._pretrained_feature_map_shape
                ),
                self._proposals_ph: np.random.randint(
                    low=0,
                    high=np.amin(self._image_shape),
                    size=self._proposals_shape,
                ),
                self._image_shape_ph: self._image_shape,
                self._gt_boxes_ph: np.random.randint(
                    low=0,
                    high=np.amin(self._image_shape),
                    size=self._gt_boxes_shape,
                ),
            }
        )
        # Assertions
        self.assertLessEqual(
            prediction_dict['target']['cls'][
                prediction_dict['target']['cls'] >= 0
            ].shape[0],
            self._config.target.minibatch_size,
        )

    def testNumberOfObjects(self):
        """Tests we're not returning more objects than we get proposals.
        """

        rcnn_net = self._shared_model(
            self._pretrained_feature_map_ph, self._proposals_ph,
            self._image_shape_ph, self._base_network
        )

        prediction_dict = self._run_net_with_feed_dict(
            rcnn_net,
            feed_dict={
                self._pretrained_feature_map_ph: np.random.rand(
                    *self._pretrained_feature_map_shape
                ),
                self._proposals_ph: np.random.randint(
                    0,
                    high=np.amin(self._image_shape),
                    size=self._proposals_shape,
                ),
                self._image_shape_ph: self._image_shape,
            }
        )
        # Assertions
        self.assertLessEqual(
            prediction_dict['objects'].shape[0],
            self._num_proposals
        )

    def testLoss(self):
        """Tests we're computing loss correctly.

        In particular, we're testing whether computing a perfect score when we
        have to.
        """

        # Generate placeholders and loss_graph
        cls_score_shape = (self._num_proposals, self._num_classes + 1)
        cls_score_ph = tf.placeholder(
            tf.float32,
            cls_score_shape
        )

        cls_prob_shape = (self._num_proposals, self._num_classes + 1)
        cls_prob_ph = tf.placeholder(
            tf.float32,
            cls_prob_shape
        )

        cls_target_shape = (self._num_proposals,)
        cls_target_ph = tf.placeholder(
            tf.float32,
            cls_target_shape
        )

        bbox_offsets_shape = (self._num_proposals, self._num_classes * 4)
        bbox_offsets_ph = tf.placeholder(
            tf.float32,
            bbox_offsets_shape
        )

        bbox_offsets_target_shape = (self._num_proposals, 4)
        bbox_offsets_target_ph = tf.placeholder(
            tf.float32,
            bbox_offsets_target_shape
        )

        loss_graph = self._shared_model.loss({
            'rcnn': {
                'cls_score': cls_score_ph,
                'cls_prob': cls_prob_ph,
                'bbox_offsets': bbox_offsets_ph,
            },
            'target': {
                'cls': cls_target_ph,
                'bbox_offsets': bbox_offsets_target_ph,
            }
        })

        # Generate values that ensure a perfect score
        # We first initialize all our values to zero.
        cls_score = np.zeros(cls_score_shape, dtype=np.float32)
        cls_prob = np.zeros(cls_prob_shape, dtype=np.float32)
        cls_target = np.zeros(cls_target_shape, dtype=np.float32)
        bbox_offsets = np.zeros(bbox_offsets_shape, dtype=np.float32)
        bbox_offsets_target = np.zeros(
            bbox_offsets_target_shape,
            dtype=np.float32
        )
        for i in range(self._num_proposals):
            this_class = np.random.randint(low=1, high=self._num_classes + 1)

            cls_score[i][this_class] = self._high_score
            cls_prob[i][this_class] = 1.
            cls_target[i] = this_class

            # Find out where in the axis 1 in bbox_offsets we should
            # put the offsets, because the shape is
            # (num_proposals, num_classes * 4), and we're using
            # 1-indexed classes.
            class_place = (this_class - 1) * 4
            for j in range(4):
                this_coord = np.random.randint(
                    low=0,
                    high=np.amax(self._image_shape)
                )

                bbox_offsets[i][class_place + j] = this_coord
                bbox_offsets_target[i][j] = this_coord
        # Now get the loss dict using the values we just generated.
        loss_dict = self._run_net_with_feed_dict(
            loss_graph,
            feed_dict={
                cls_score_ph: cls_score,
                cls_prob_ph: cls_prob,
                cls_target_ph: cls_target,
                bbox_offsets_ph: bbox_offsets,
                bbox_offsets_target_ph: bbox_offsets_target,
            }
        )
        # Assertions
        self.assertAlmostEqual(
            loss_dict['rcnn_cls_loss'], 0,
            delta=self._equality_delta
        )
        self.assertAlmostEqual(
            loss_dict['rcnn_reg_loss'], 0,
            delta=self._equality_delta
        )


if __name__ == '__main__':
    tf.test.main()
