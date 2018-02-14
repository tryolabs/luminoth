import os
import json
import click
import tensorflow as tf

from flask import Flask, jsonify, request, render_template
from PIL import Image

from luminoth.utils.config import get_config
from luminoth.utils.predicting import PredictorNetwork


app = Flask(__name__)


def get_image():
    image = request.files.get('image')
    if not image:
        return None

    image = Image.open(image.stream).convert('RGB')
    return image


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/<model_name>/predict', methods=['GET', 'POST'])
def predict(model_name):
    if request.method == 'GET':
        return jsonify(error='Use POST method to send image.'), 400

    image_array = get_image()
    if image_array is None:
        return jsonify(error='Missing image.'), 400

    total_predictions = request.args.get('total')
    if total_predictions is not None:
        try:
            total_predictions = int(total_predictions)
        except ValueError:
            total_predictions = None

    prediction = PREDICTOR_NETWORK.predict_image(image_array, total_predictions)
    return jsonify(prediction)


@click.command(help='Start basic web application.')
@click.option('config_files', '--config', '-c', required=True, multiple=True, help='Config to use.')  # noqa
@click.option('--host', default='127.0.0.1', help='Hostname to listen on. Set this to "0.0.0.0" to have the server available externally.')  # noqa
@click.option('--port', default=5000, help='Port to listen to.')
@click.option('--debug', is_flag=True, help='Set debug level logging.')
def web(config_files, host, port, debug):
    if debug:
        tf.logging.set_verbosity(tf.logging.DEBUG)
    else:
        tf.logging.set_verbosity(tf.logging.INFO)

    config = get_config(config_files)

    # Bounding boxes will be filtered by frontend (using slider), so we set
    # a low threshold.
    config.model.rcnn.proposals.min_prob_threshold = 0.01

    if config.dataset.dir:
        # Gets the names of the classes
        classes_file = os.path.join(config.dataset.dir, 'classes.json')
        if tf.gfile.Exists(classes_file):
            config['class_labels'] = json.load(
                tf.gfile.GFile(classes_file))

    # Initialize model
    global PREDICTOR_NETWORK
    PREDICTOR_NETWORK = PredictorNetwork(config_files)

    app.run(host=host, port=port, debug=debug)
