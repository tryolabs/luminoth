import click
import time
import numpy as np
import tensorflow as tf

from flask import Flask, jsonify, request, render_template
from luminoth.models import get_model
from PIL import Image

app = Flask(__name__)

LOADED_MODELS = {}


def get_image():
    image = request.files.get('image')
    if not image:
        return None

    image = Image.open(image.stream)
    return image


def resize_image(image, min_size, max_size):
    min_dimension = min(image.height, image.width)
    upscale = max(min_size / min_dimension, 1.)

    max_dimension = max(image.height, image.width)
    downscale = min(max_size / max_dimension, 1.)

    new_width = int(upscale * downscale * image.width)
    new_height = int(upscale * downscale * image.height)

    image = image.resize((new_width, new_height))
    image_array = np.array(image)[:, :, :3]  # TODO Read RGB
    image_array = np.expand_dims(image_array, axis=0)
    return image_array, upscale * downscale


def get_prediction(model_name, image, checkpoint_file=None):
    model_class = get_model(model_name)
    if model_name in LOADED_MODELS:
        image_tensor, output, graph, session = LOADED_MODELS[model_name]
    else:
        graph = tf.Graph()
        session = tf.Session(graph=graph)

        with graph.as_default():
            image_tensor = tf.placeholder(tf.float32, (1, None, None, 3))
            model = model_class(model_class.base_config)
            output = model(image_tensor)
            if checkpoint_file:
                saver = tf.train.Saver(sharded=True, allow_empty=True)
                saver.restore(session, checkpoint_file)
            else:
                init_op = tf.group(
                    tf.global_variables_initializer(),
                    tf.local_variables_initializer()
                )
                session.run(init_op)

        LOADED_MODELS[model_name] = (image_tensor, output, graph, session)

    classification_prediction = output['classification_prediction']
    objects_tf = classification_prediction['objects']
    objects_labels_tf = classification_prediction['labels']
    objects_labels_prob_tf = classification_prediction['probs']
    image_resize_config = model_class.base_config.dataset.image_preprocessing
    image_array, scale_factor = resize_image(
        image, image_resize_config.min_size, image_resize_config.max_size
    )

    start_time = time.time()
    objects, objects_labels, objects_labels_prob = session.run([
        objects_tf, objects_labels_tf, objects_labels_prob_tf
    ], feed_dict={
        image_tensor: image_array
    })
    end_time = time.time()

    return {
        'objects': objects.tolist(),
        'objects_labels': objects_labels.tolist(),
        'objects_labels_prob': objects_labels_prob.tolist(),
        'inference_time': end_time - start_time,
        'scale_factor': scale_factor,
    }


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
        model_name, image_array, app.config['checkpoint_file'])

    return jsonify(pred)


@click.command(help='Start basic web application.')
@click.option('--checkpoint-file')
def web(checkpoint_file):
    app.config['checkpoint_file'] = checkpoint_file
    app.run(debug=True)
