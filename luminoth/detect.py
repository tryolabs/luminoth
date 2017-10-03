import click
import os

from luminoth.utils.predict import get_prediction
from PIL import Image, ImageDraw


@click.command(help='Detects objects in a image.')
@click.argument('image-path')
@click.option('--checkpoint-file')
@click.option('--classes-file')
def detect(image_path, checkpoint_file, classes_file):
    image = Image.open(image_path)
    results = get_prediction(
        'fasterrcnn', image, checkpoint_file, classes_file)
    print('{} objects detected'.format(len(results['objects'])))

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
    if checkpoint_file:
        save_path = checkpoint_file + '_pred_' + os.path.basename(image_path)
    else:
        save_path = 'pred_' + os.path.basename(image_path)
    print('Saving image with bounding boxes in {}'.format(save_path))
    image.save(save_path)
