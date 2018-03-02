import click
import tensorflow as tf

from flask import Flask, jsonify, request, render_template
from threading import Thread
from PIL import Image

from luminoth.tools.checkpoint import get_checkpoint_config
from luminoth.utils.config import get_config, override_config_params
from luminoth.utils.predicting import PredictorNetwork


app = Flask(__name__)


def get_image():
    image = request.files.get('image')
    if not image:
        raise ValueError

    image = Image.open(image.stream).convert('RGB')
    return image


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/<model_name>/predict', methods=['GET', 'POST'])
def predict(model_name):
    if request.method == 'GET':
        return jsonify(error='Use POST method to send image.'), 400

    try:
        image_array = get_image()
    except ValueError:
        return jsonify(error='Missing image'), 400
    except OSError:
        return jsonify(error='Incompatible file type'), 400

    total_predictions = request.args.get('total')
    if total_predictions is not None:
        try:
            total_predictions = int(total_predictions)
        except ValueError:
            total_predictions = None

    NETWORK_START_THREAD.join()
    prediction = PREDICTOR_NETWORK.predict_image(
        image_array, total_predictions
    )
    return jsonify(prediction)


def start_network(config):
    global PREDICTOR_NETWORK
    PREDICTOR_NETWORK = PredictorNetwork(config)


@click.command(help='Start basic web application.')
@click.option('config_files', '--config', '-c', multiple=True, help='Config to use.')  # noqa
@click.option('--checkpoint', help='Checkpoint to use.')
@click.option('override_params', '--override', '-o', multiple=True, help='Override model config params.')  # noqa
@click.option('--host', default='127.0.0.1', help='Hostname to listen on. Set this to "0.0.0.0" to have the server available externally.')  # noqa
@click.option('--port', default=5000, help='Port to listen to.')
@click.option('--debug', is_flag=True, help='Set debug level logging.')
def web(config_files, checkpoint, override_params, host, port, debug):
    if debug:
        tf.logging.set_verbosity(tf.logging.DEBUG)
    else:
        tf.logging.set_verbosity(tf.logging.INFO)

    if checkpoint:
        config = get_checkpoint_config(checkpoint)
    elif config_files:
        config = get_config(config_files)
    else:
        click.echo('You must specify either a checkpoint or a config file.')
        return

    if override_params:
        config = override_config_params(config, override_params)

    # Bounding boxes will be filtered by frontend (using slider), so we set a
    # low threshold.
    config.model.rcnn.proposals.min_prob_threshold = 0.01

    # Initialize model
    global NETWORK_START_THREAD
    NETWORK_START_THREAD = Thread(target=start_network, args=(config,))
    NETWORK_START_THREAD.start()

    app.run(host=host, port=port, debug=debug)
