import tensorflow as tf

from tensorflow.python.training.summary_io import SummaryWriterCache


class VarVisHook(tf.train.SessionRunHook):

    def __init__(self, every_n_steps=None, every_n_secs=None, mode=None,
                 output_dir=None, vars_summary=None):
        super(VarVisHook, self).__init__()

        if (every_n_secs is None) == (every_n_steps is None):
            raise ValueError(
                'Only one of "every_n_secs" and "every_n_steps" must be '
                'provided.'
            )

        if output_dir is None:
            tf.logging.warning(
                '`output_dir` not provided, VarVisHook is not saving '
                'summaries.'
            )

        self._timer = tf.train.SecondOrStepTimer(
            every_steps=every_n_steps,
            every_secs=every_n_secs
        )

        self._mode = mode
        self._output_dir = output_dir
        self._summary_writer = None
        self._vars_summary = vars_summary

        tf.logging.info('VarVisHook was created with mode = "{}"'.format(mode))

    def begin(self):
        if self._output_dir:
            self._summary_writer = SummaryWriterCache.get(self._output_dir)

        self._next_step = None
        self._global_step = tf.train.get_global_step()
        if self._global_step is None:
            raise RuntimeError('Global step must be created for VarVisHook.')

    def before_run(self, run_context):
        fetches = {
            'global_step': self._global_step,
        }

        self._write_summaries = (
            self._next_step is None or
            self._timer.should_trigger_for_step(self._next_step)
        )

        if self._write_summaries:
            fetches['summary'] = self._vars_summary[self._mode]

        return tf.train.SessionRunArgs(fetches)

    def after_run(self, run_context, run_values):
        results = run_values.results
        global_step = results.get('global_step')

        if self._write_summaries:
            self._timer.update_last_triggered_step(global_step)
            summary = results.get('summary')
            if summary is not None:
                if self._summary_writer is not None:
                    self._summary_writer.add_summary(summary, global_step)

        self._next_step = global_step + 1

    def end(self, session=None):
        if self._summary_writer:
            self._summary_writer.flush()
