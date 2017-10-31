import click
import json
import os
import tensorflow as tf

from PIL import Image, ImageDraw

from luminoth.utils.predicting import get_prediction, load_config


@click.command(help='Obtain a model\'s predictions in a given image.')
@click.argument('image-path')
@click.option('config_files', '--config', '-c', required=True, multiple=True, help='Config to use.')  # noqa
@click.option('--model-type', default='fasterrcnn')
@click.option('--save/--no-save', default=False,
              help='Save the image with the prediction of the model')
def predict(image_path, config_files, model_type, save):
    image = Image.open(image_path)
    results = get_prediction(model_type, image, config_files)
    tf.logging.info('{} objects detected'.format(len(results['objects'])))

    # Define the save path
    config = load_config(config_files)
    run_name = config.get('train', {}).get('run_name')

    if run_name:
        save_path = run_name + '_pred_' + os.path.basename(image_path)
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
