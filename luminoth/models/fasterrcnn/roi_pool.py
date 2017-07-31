import sonnet as snt
import tensorflow as tf

CROP = 'crop'
ROI_POOLING = 'roi_pooling'


class ROIPoolingLayer(snt.AbstractModule):
    """ROIPoolingLayer which applies ROI pooling (or tf.crop_and_resize)"""
    def __init__(self, config, debug=False, name='roi_pooling'):
        super(ROIPoolingLayer, self).__init__(name=name)
        self._pooling_mode = config.pooling_mode.lower()
        self._pooled_width = config.pooled_width
        self._pooled_height = config.pooled_height
        self._pooled_padding = config.padding
        self._debug = debug

    def _get_bboxes(self, roi_proposals, im_shape):
        """
        Get normalized coordinates for RoIs (between 0 and 1 for easy cropping)
        in TF order (y1, x1, y2, x2)
        """
        with tf.name_scope('get_bboxes'):
            im_shape = tf.cast(im_shape, tf.float32)

            _, x1, y1, x2, y2 = tf.unstack(
                roi_proposals, axis=1
            )

            x1 = x1 / im_shape[1]
            y1 = y1 / im_shape[0]
            x2 = x2 / im_shape[1]
            y2 = y2 / im_shape[0]

            # Won't be backpropagated to rois anyway, but to save time TODO: Remove
            bboxes = tf.stack([y1, x1, y2, x2], axis=1)

            return bboxes

    def _roi_crop(self, roi_proposals, pretrained, im_shape):
        bboxes = self._get_bboxes(roi_proposals, im_shape)
        bboxes_shape = tf.shape(bboxes)
        batch_ids = tf.zeros((bboxes_shape[0], ), dtype=tf.int32)
        crops = tf.image.crop_and_resize(
            pretrained, bboxes, batch_ids,
            [self._pooled_width * 2, self._pooled_height * 2], name="crops"
        )

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
            prediction_dict['pretrained'] = pretrained

        return prediction_dict

    def _roi_pooling(self, roi_proposals, pretrained, im_shape):
        raise NotImplemented()

    def _build(self, roi_proposals, pretrained, im_shape):
        if self._pooling_mode == CROP:
            return self._roi_crop(roi_proposals, pretrained, im_shape)
        elif self._pooling_mode == ROI_POOLING:
            return self._roi_pooling(roi_proposals, pretrained, im_shape)
        else:
            raise NotImplemented(
                'Pooling mode {} does not exist.'.format(self._pooling_mode))
