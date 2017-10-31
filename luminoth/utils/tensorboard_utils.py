import logging
import tensorflow as tf
import tensorboard.main as tb
import threading

from tensorboard.plugins.audio import audio_plugin
from tensorboard.plugins.core import core_plugin
from tensorboard.plugins.distribution import distributions_plugin
from tensorboard.plugins.graph import graphs_plugin
from tensorboard.plugins.histogram import histograms_plugin
from tensorboard.plugins.image import images_plugin
from tensorboard.plugins.profile import profile_plugin
from tensorboard.plugins.projector import projector_plugin
from tensorboard.plugins.scalar import scalars_plugin
from tensorboard.plugins.text import text_plugin


def set_up(logdir):
    plugins = [
        core_plugin.CorePlugin,
        scalars_plugin.ScalarsPlugin,
        images_plugin.ImagesPlugin,
        audio_plugin.AudioPlugin,
        graphs_plugin.GraphsPlugin,
        distributions_plugin.DistributionsPlugin,
        histograms_plugin.HistogramsPlugin,
        projector_plugin.ProjectorPlugin,
        text_plugin.TextPlugin,
        profile_plugin.ProfilePlugin,
    ]
    werkzeug_logger = logging.getLogger('werkzeug')
    werkzeug_logger.setLevel(tf.logging.ERROR)
    tf.flags.FLAGS.logdir = logdir
    tf.logging.set_verbosity(tf.logging.ERROR)
    app = tb.create_tb_app(plugins)
    tb.run_simple_server(app)


def tensorboard_start(logdir):
    thread = threading.Thread(target=set_up, args=(logdir,))
    thread.daemon = True
    thread.start()
