import click

from flask import Flask, jsonify, request, render_template
from luminoth.utils.predict import get_prediction
from PIL import Image

app = Flask(__name__)


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

    pred = get_prediction(
        model_name, image_array, app.config['checkpoint_file'],
        app.config['classes_file']
    )

    return jsonify(pred)


@click.command(help='Start basic web application.')
@click.option('--checkpoint-file')
@click.option('--classes-file')
def web(checkpoint_file, classes_file):
    app.config['checkpoint_file'] = checkpoint_file
    app.config['classes_file'] = classes_file
    app.run(debug=True)
