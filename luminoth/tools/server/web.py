import os
import json
import click
import tensorflow as tf

from flask import Flask, jsonify, request, render_template
from PIL import Image

from luminoth.utils.config import get_config
from luminoth.utils.predicting import get_prediction


app = Flask(__name__)

LOADED_MODELS = {}


def get_image():
    image = request.files.get('image')
    if not image:
        return None

    image = Image.open(image.stream)
    return image


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/<model_name>/predict', methods=['GET', 'POST'])
def predict(model_name):
    if request.method == 'GET':
        return jsonify(error='Use POST method to send image.')

    image_array = get_image()
    if image_array is None:
        return jsonify(error='Missing image.')

    config = app.config['config']
    class_labels = app.config.get('class_labels')

    if model_name in LOADED_MODELS:
        image_tensor, fetches, session = LOADED_MODELS[model_name]
        pred = get_prediction(
            image_array, config, session=session, fetches=fetches,
            image_tensor=image_tensor, class_labels=class_labels
        )
    else:
        pred = get_prediction(
            image_array, config, class_labels=class_labels,
            return_tf_vars=True
        )
        LOADED_MODELS[model_name] = (
            pred['image_tensor'], pred['fetches'], pred['session']
        )

        del pred['image_tensor'], pred['fetches'], pred['session']
    return jsonify(pred)


@click.command(help='Start basic web application.')
@click.option('config_files', '--config', '-c', required=True, multiple=True, help='Config to use.')  # noqa
def web(config_files):
    config = get_config(config_files)
    app.config['config'] = config
    if config.dataset.dir:
        # Gets the names of the classes
        classes_file = os.path.join(config.dataset.dir, 'classes.json')
        if tf.gfile.Exists(classes_file):
            app.config['class_labels'] = json.load(
                tf.gfile.GFile(classes_file))

    app.run(debug=True)
