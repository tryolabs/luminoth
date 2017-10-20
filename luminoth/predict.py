import click
import easydict
import tensorflow as tf
import os
import yaml
import json

from luminoth.utils.predicting import get_prediction
from PIL import Image, ImageDraw


@click.command(help='Obtain a model\'s predictions in a given image.')
@click.argument('image-path')
@click.option('--config-file')
@click.option('--model-type', default='fasterrcnn')
@click.option('--save/--no-save', default=False,
              help='Save the image with the prediction of the model')
def predict(image_path, config_file, model_type, save):
    image = Image.open(image_path)
    results = get_prediction(model_type, image, config_file)
    tf.logging.info('{} objects detected'.format(len(results['objects'])))

    # Define the save path
    config = easydict.EasyDict(yaml.load(tf.gfile.GFile(config_file)))
    if config.train.job_dir and config.train.run_name:
        save_path = (config.train.run_name + '_pred_' +
                     os.path.basename(image_path))
    else:
        save_path = 'pred_' + os.path.basename(image_path)

    with open(save_path + '.json', 'w') as outfile:
        json.dump(results, outfile)

    if save:
        # Draw bounding boxes
        draw = ImageDraw.Draw(image)
        scale = results['scale_factor']
        for ind, bbox in enumerate(results['objects']):
            bbox_res = [i / scale for i in bbox]
            draw.rectangle(bbox_res, outline='red')
            draw.text(
                bbox_res[:2],
                ' '.join((str(results['objects_labels'][ind]),
                         str(results['objects_labels_prob'][ind])))
            )

        # Save the image
        tf.logging.info('Saving image with bounding boxes in {}'
                        .format(save_path))
        image.save(save_path)
