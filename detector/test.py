import click
import tensorflow as tf
import numpy as np
import io

from scipy.misc import imread, toimage

from .detector import inference, IMAGE_SIZE
from .voc import read_classes
from .utils.imgcat import imgcat

@click.command()
@click.option('--model-dir', required=True)
@click.option('--dataset-dir', default='datasets/voc/')
@click.option('--image-file', required=True)
@click.option('--print-inline', is_flag=True, default=False)
def test(model_dir, dataset_dir, image_file, print_inline):
    # Read model
    ckpt = tf.train.get_checkpoint_state(model_dir)
    if not ckpt or not ckpt.model_checkpoint_path:
        print(f'error: no mode in {model_dir}')
        return
    print(f'model found in "{model_dir}"')

    classes = read_classes(dataset_dir)
    print(f'{len(classes)} classes in dataset')

    init_op = tf.group(
        tf.global_variables_initializer(),
        tf.local_variables_initializer()
    )

    image_raw = np.array(imread(image_file))
    image = tf.placeholder(
        tf.float32
    )
    # resized_image = tf.image.resize_images(
    #     image, [IMAGE_SIZE, IMAGE_SIZE]
    # )
    resized_image = tf.image.resize_image_with_crop_or_pad(
        image, IMAGE_SIZE, IMAGE_SIZE
    )
    resized_image.set_shape([IMAGE_SIZE, IMAGE_SIZE, 3])
    std_image = tf.image.per_image_standardization(resized_image)
    images = tf.expand_dims(std_image, 0)

    y_pred, end_points = inference(images, is_training=False)
    y_prob = tf.nn.softmax(y_pred)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        # Run the initializer, then the rest.
        print('starting session')
        sess.run(init_op)

        # Restore the checkpoint.
        print(f'restoring session from "{ckpt.model_checkpoint_path}"')
        saver.restore(sess, ckpt.model_checkpoint_path)
        step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        print('running prediction on images')

        y_pred_values = sess.run(y_prob, feed_dict={
            image: image_raw
        }).flatten()

        class_id = y_pred_values.argmax()

        imgcat(open(image_file, 'rb').read())

        resized_image_bytes = io.BytesIO()
        toimage(sess.run(resized_image, feed_dict={image: image_raw})).save(resized_image_bytes, format='PNG')
        resized_image_bytes = resized_image_bytes.getvalue()
        imgcat(resized_image_bytes)
        print(f'class for image is "{classes[class_id]}", prob: {y_pred_values[class_id]}')
        print(f'y_pred_values = {y_pred_values}')
        import ipdb; ipdb.set_trace()
        print('closing session')

