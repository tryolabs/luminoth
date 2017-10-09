import click
import easydict
import tensorflow as tf
import os
import yaml

from luminoth.utils.predict import get_prediction
from PIL import Image, ImageDraw


@click.command(help='Detects objects in a image.')
@click.argument('image-path')
@click.option('--config-file')
@click.option('--model-name', default='fasterrcnn')
def detect(image_path, config_file, model_name):
    image = Image.open(image_path)
    results = get_prediction(model_name, image, config_file)
    tf.logging.info('{} objects detected'.format(len(results['objects'])))
    print(results)
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
    config = easydict.EasyDict(yaml.load(tf.gfile.GFile(config_file)))
    print(config.train)
    if config.train.job_dir and config.train.run_name:
        save_path = config.train.run_name + '_pred_' + os.path.basename(image_path)
    else:
        save_path = 'pred_' + os.path.basename(image_path)
    tf.logging.info('Saving image with bounding boxes in {}'.format(save_path))
    image.save(save_path)
