import click

from flask import Flask, jsonify, request, render_template
from luminoth.utils.predict import get_prediction
from PIL import Image

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

    if model_name in LOADED_MODELS:
        image_tensor, prediction_dict, session = LOADED_MODELS[model_name]
        pred = get_prediction(
            model_name, image_array, app.config['config_file'],
            session=session, prediction_dict=prediction_dict,
            image_tensor=image_tensor
        )

    else:
        pred = get_prediction(model_name, image_array,
                              app.config['config_file'], return_tf_vars=True)
        LOADED_MODELS[model_name] = (pred['image_tensor'],
                                     pred['prediction_dict'],
                                     pred['session'])

        del pred['image_tensor'], pred['prediction_dict'], pred['session']
    return jsonify(pred)


@click.command(help='Start basic web application.')
@click.option('--config-file')
def web(config_file):
    app.config['config_file'] = config_file
    app.run(debug=True)
