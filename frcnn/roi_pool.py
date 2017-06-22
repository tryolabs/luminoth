import sonnet as snt
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np


CROP = 'crop'
ROI_POOLING = 'roi_pooling'


class ROIPoolingLayer(snt.AbstractModule):
    """ROIPoolingLayer which applies ROI pooling (or tf.crop_and_resize)"""
    def __init__(self, pooling_mode=CROP, pooled_width=7, pooled_height=7,
                 spatial_scale=1./16, feat_stride=[16], name='roi_pooling'):
        super(ROIPoolingLayer, self).__init__(name=name)
        self._pooling_mode = pooling_mode
        self._pooled_width = pooled_width
        self._pooled_height = pooled_height
        self._spatial_scale = spatial_scale
        self._feat_stride = feat_stride

    def _get_bboxes(self, roi_proposals, pretrained):
        """
        Get normalized coordinates for RoIs (betweetn 0 and 1 for easy cropping)
        """
        pretrained_shape = tf.shape(pretrained)
        height = (tf.to_float(pretrained_shape[1]) - 1.) * np.float32(self._feat_stride[0])
        width = (tf.to_float(pretrained_shape[2]) - 1.) * np.float32(self._feat_stride[0])

        x1 = tf.slice(roi_proposals, [0, 1], [-1, 1], name="x1") / width
        y1 = tf.slice(roi_proposals, [0, 2], [-1, 1], name="y1") / height
        x2 = tf.slice(roi_proposals, [0, 3], [-1, 1], name="x2") / width
        y2 = tf.slice(roi_proposals, [0, 4], [-1, 1], name="y2") / height

        # Won't be backpropagated to rois anyway, but to save time TODO: Remove?
        bboxes = tf.stop_gradient(tf.concat([y1, x1, y2, x2], axis=1))

        return bboxes

    def _roi_crop(self, roi_proposals, pretrained):

        bboxes = self._get_bboxes(roi_proposals, pretrained)
        # TODO: Why?!!?
        # batch_ids = tf.squeeze(tf.slice(roi_proposals, [0, 0], [-1, 1], name="batch_id"), [1])
        bboxes_shape = tf.shape(bboxes)
        batch_ids = tf.zeros((bboxes_shape[0], ), dtype=tf.int32)
        crops = tf.image.crop_and_resize(
            pretrained, bboxes, batch_ids,
            [self._pooled_width * 2, self._pooled_height * 2], name="crops"
        )

        return slim.max_pool2d(crops, [2, 2], stride=2)


    def _roi_pooling(self, roi_proposals, pretrained):
        raise NotImplemented()

    def _build(self, roi_proposals, pretrained):
        if self._pooling_mode == CROP:
            return self._roi_crop(roi_proposals, pretrained)
        elif self._pooling_mode == ROI_POOLING:
            return self._roi_pooling(roi_proposals, pretrained)
        else:
            raise NotImplemented('Pooling mode {} does not exist.'.format(self._pooling_mode))
