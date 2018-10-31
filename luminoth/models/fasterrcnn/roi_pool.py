import sonnet as snt
import tensorflow as tf

# Types of RoI "pooling"
CROP = 'crop'
ROI_POOLING = 'roi_pooling'


class ROIPoolingLayer(snt.AbstractModule):
    """ROIPoolingLayer applies ROI Pooling (or tf.crop_and_resize).

    RoI pooling or RoI extraction is used to extract fixed size features from a
    variable sized feature map using variabled sized bounding boxes. Since we
    have proposals of different shapes and sizes, we need a way to transform
    them into a fixed size Tensor for using FC layers.

    There are two basic ways to do this, the original one in the FasterRCNN's
    paper is RoI Pooling, which as the name suggests, it maxpools directly from
    the region of interest, or proposal, into a fixed size Tensor.

    The alternative way uses TensorFlow's image utility operation called,
    `crop_and_resize` which first crops an Tensor using a normalized proposal,
    and then applies extrapolation to resize it to the desired size,
    generating a fixed size Tensor.

    Since there isn't a std support implemenation of RoIPooling, we apply the
    easier but still proven alternatve way.
    """
    def __init__(self, config, debug=False, name='roi_pooling'):
        super(ROIPoolingLayer, self).__init__(name=name)
        self._pooling_mode = config.pooling_mode.lower()
        self._pooled_width = config.pooled_width
        self._pooled_height = config.pooled_height
        self._pooled_padding = config.padding
        self._debug = debug

    def _get_bboxes(self, roi_proposals, im_shape):
        """
        Gets normalized coordinates for RoIs (between 0 and 1 for cropping)
        in TensorFlow's order (y1, x1, y2, x2).

        Args:
            roi_proposals: A Tensor with the bounding boxes of shape
                (total_proposals, 5), where the values for each proposal are
                (x_min, y_min, x_max, y_max).
            im_shape: A Tensor with the shape of the image (height, width).

        Returns:
            bboxes: A Tensor with normalized bounding boxes in TensorFlow's
                format order. Its should is (total_proposals, 4).
        """
        with tf.name_scope('get_bboxes'):
            im_shape = tf.cast(im_shape, tf.float32)

            x1, y1, x2, y2 = tf.unstack(
                roi_proposals, axis=1
            )

            x1 = x1 / im_shape[1]
            y1 = y1 / im_shape[0]
            x2 = x2 / im_shape[1]
            y2 = y2 / im_shape[0]

            bboxes = tf.stack([y1, x1, y2, x2], axis=1)

            return bboxes

    def _roi_crop(self, roi_proposals, conv_feature_map, im_shape):
        # Get normalized bounding boxes.
        bboxes = self._get_bboxes(roi_proposals, im_shape)
        # Generate fake batch ids
        bboxes_shape = tf.shape(bboxes)
        batch_ids = tf.zeros((bboxes_shape[0], ), dtype=tf.int32)
        # Apply crop and resize with extracting a crop double the desired size.
        crops = tf.image.crop_and_resize(
            conv_feature_map, bboxes, batch_ids,
            [self._pooled_width * 2, self._pooled_height * 2], name="crops"
        )

        # Applies max pool with [2,2] kernel to reduce the crops to half the
        # size, and thus having the desired output.
        prediction_dict = {
            'roi_pool': tf.nn.max_pool(
                crops, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                padding=self._pooled_padding
            ),
        }

        if self._debug:
            prediction_dict['bboxes'] = bboxes
            prediction_dict['crops'] = crops
            prediction_dict['batch_ids'] = batch_ids
            prediction_dict['conv_feature_map'] = conv_feature_map

        return prediction_dict

    def _roi_pooling(self, roi_proposals, conv_feature_map, im_shape):
        raise NotImplementedError()

    def _build(self, roi_proposals, conv_feature_map, im_shape):
        if self._pooling_mode == CROP:
            return self._roi_crop(roi_proposals, conv_feature_map, im_shape)
        elif self._pooling_mode == ROI_POOLING:
            return self._roi_pooling(roi_proposals, conv_feature_map, im_shape)
        else:
            raise NotImplementedError(
                'Pooling mode {} does not exist.'.format(self._pooling_mode))
