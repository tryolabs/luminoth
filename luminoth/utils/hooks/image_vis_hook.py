import tensorflow as tf

from tensorflow.python.training.summary_io import SummaryWriterCache
from luminoth.utils.image_vis import image_vis_summaries


class ImageVisHook(tf.train.SessionRunHook):
    def __init__(self, prediction_dict, image, with_rcnn=True,
                 gt_bboxes=None, every_n_steps=None, every_n_secs=None,
                 output_dir=None, summary_writer=None,
                 image_visualization_mode=None):
        super(ImageVisHook, self).__init__()
        if (every_n_secs is None) == (every_n_steps is None):
            raise ValueError(
                'Only one of "every_n_secs" and "every_n_steps" must be '
                'provided.')
        if output_dir is None and summary_writer is None:
            tf.logging.warning(
                'ImageVisHook is not saving summaries. One of "output_dir" '
                'and "summary_writer" must be provided')
        self._timer = tf.train.SecondOrStepTimer(
            every_steps=every_n_steps, every_secs=every_n_secs)

        self._prediction_dict = prediction_dict
        self._with_rcnn = with_rcnn
        self._output_dir = output_dir
        self._summary_writer = summary_writer
        self._image_visualization_mode = image_visualization_mode
        self._image = image
        self._gt_bboxes = gt_bboxes

        tf.logging.info('ImageVisHook was created with mode = "{}"'.format(
            image_visualization_mode
        ))

    def begin(self):
        if self._summary_writer is None and self._output_dir:
            self._summary_writer = SummaryWriterCache.get(self._output_dir)
        self._next_step = None
        self._global_step = tf.train.get_global_step()
        if self._global_step is None:
            raise RuntimeError('Global step must be created for ImageVisHook.')

    def before_run(self, run_context):

        fetches = {'global_step': self._global_step}
        self._draw_images = (
            self._next_step is None or
            self._timer.should_trigger_for_step(self._next_step)
        )

        if self._draw_images:
            fetches['prediction_dict'] = self._prediction_dict
            fetches['image'] = self._image
            if self._gt_bboxes is not None:
                fetches['gt_bboxes'] = self._gt_bboxes

        return tf.train.SessionRunArgs(fetches)

    def after_run(self, run_context, run_values):
        results = run_values.results
        global_step = results.get('global_step')

        if self._draw_images:
            self._timer.update_last_triggered_step(global_step)
            prediction_dict = results.get('prediction_dict')
            if prediction_dict is not None:
                summaries = image_vis_summaries(
                    prediction_dict, with_rcnn=self._with_rcnn,
                    image_visualization_mode=self._image_visualization_mode,
                    image=results.get('image'),
                    gt_bboxes=results.get('gt_bboxes')
                )
                if self._summary_writer is not None:
                    for summary in summaries:
                        self._summary_writer.add_summary(summary, global_step)

        self._next_step = global_step + 1

    def end(self, session=None):
        if self._summary_writer:
            self._summary_writer.flush()
