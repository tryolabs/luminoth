"""Provides various visualization-specific functions."""
import numpy as np
import sys

from PIL import Image, ImageDraw, ImageFont


def get_font():
    """Attempts to retrieve a reasonably-looking TTF font from the system.

    We don't make much of an effort, but it's what we can reasonably do without
    incorporating additional dependencies for this task.
    """
    if sys.platform == 'win32':
        font_names = ['Arial']
    elif sys.platform in ['linux', 'linux2']:
        font_names = ['DejaVuSans-Bold', 'DroidSans-Bold']
    elif sys.platform == 'darwin':
        font_names = ['Menlo', 'Helvetica']

    font = None
    for font_name in font_names:
        try:
            font = ImageFont.truetype(font_name)
            break
        except IOError:
            continue

    return font


SYSTEM_FONT = get_font()


def hex_to_rgb(x):
    """Turns a color hex representation into a tuple representation."""
    return tuple([int(x[i:i + 2], 16) for i in (0, 2, 4)])


def build_colormap():
    """Builds a colormap function that maps labels to colors.

    Returns:
        Function that receives a label and returns a color tuple `(R, G, B)`
        for said label.
    """
    # Build the 10-color palette to be used for all classes. The following are
    # the hex-codes for said colors (taken the default 10-categorical d3 color
    # palette).
    palette = (
        '1f77b4ff7f0e2ca02cd627289467bd8c564be377c27f7f7fbcbd2217becf'
    )
    colors = [hex_to_rgb(palette[i:i + 6]) for i in range(0, len(palette), 6)]

    seen_labels = {}

    def colormap(label):
        # If label not yet seen, get the next value in the palette sequence.
        if label not in seen_labels:
            seen_labels[label] = colors[len(seen_labels) % len(colors)]

        return seen_labels[label]

    return colormap


def draw_rectangle(draw, coordinates, color, width=1, fill=30):
    """Draw a rectangle with an optional width."""
    # Add alphas to the color so we have a small overlay over the object.
    fill = color + (fill,)
    outline = color + (255,)

    # Pillow doesn't support width in rectangles, so we must emulate it with a
    # loop.
    for i in range(width):
        coords = [
            coordinates[0] - i,
            coordinates[1] - i,
            coordinates[2] + i,
            coordinates[3] + i,
        ]

        # Fill must be drawn only for the first rectangle, or the alphas will
        # add up.
        if i == 0:
            draw.rectangle(coords, fill=fill, outline=outline)
        else:
            draw.rectangle(coords, outline=outline)


def draw_label(draw, coords, label, prob, color, scale=1):
    """Draw a box with the label and probability."""
    # Attempt to get a native TTF font. If not, use the default bitmap font.
    global SYSTEM_FONT
    if SYSTEM_FONT:
        label_font = SYSTEM_FONT.font_variant(size=int(round(16 * scale)))
        prob_font = SYSTEM_FONT.font_variant(size=int(round(12 * scale)))
    else:
        label_font = ImageFont.load_default()
        prob_font = ImageFont.load_default()

    label = str(label)  # `label` may not be a string.
    prob = '({:.2f})'.format(prob)  # Turn `prob` into a string.

    # We want the probability font to be smaller, so we'll write the label in
    # two steps.
    label_w, label_h = label_font.getsize(label)
    prob_w, prob_h = prob_font.getsize(prob)

    # Get margins to manually adjust the spacing. The margin goes between each
    # segment (i.e. margin, label, margin, prob, margin).
    margin_w, margin_h = label_font.getsize('M')
    margin_w *= 0.2
    _, full_line_height = label_font.getsize('Mq')

    # Draw the background first, considering all margins and the full line
    # height.
    background_coords = [
        coords[0],
        coords[1],
        coords[0] + label_w + prob_w + 3 * margin_w,
        coords[1] + full_line_height * 1.15,
    ]
    draw.rectangle(background_coords, fill=color + (255,))

    # Then write the two pieces of text.
    draw.text([
        coords[0] + margin_w,
        coords[1],
    ], label, font=label_font)

    draw.text([
        coords[0] + label_w + 2 * margin_w,
        coords[1] + (margin_h - prob_h),
    ], prob, font=prob_font)


def vis_objects(image, objects, colormap=None, labels=True, scale=1, fill=30):
    """Visualize objects as returned by `Detector`.

    Arguments:
        image (numpy.ndarray): Image to draw the bounding boxes on.
        objects (list of dicts or dict): List of objects as returned by a
            `Detector` instance.
        colormap (function): Colormap function to use for the objects.
        labels (boolean): Whether to draw labels.
        scale (float): Scale factor for the box sizes, which will enlarge or
            shrink the width of the boxes and the fonts.
        fill (int): Integer between 0..255 to use as fill for the bounding
            boxes.

    Returns:
        A PIL image with the detected objects' bounding boxes and labels drawn.
        Can be casted to a `numpy.ndarray` by using `numpy.array` on the
        returned object.
    """
    if not isinstance(objects, list):
        objects = [objects]

    if colormap is None:
        colormap = build_colormap()

    image = Image.fromarray(image.astype(np.uint8))

    draw = ImageDraw.Draw(image, 'RGBA')
    for obj in objects:
        # TODO: Can we do image resolution-agnostic?
        color = colormap(obj['label'])

        # Cast `width` to int so it also works in Python 2
        draw_rectangle(
            draw, obj['bbox'], color, width=int(round(3 * scale)),
            fill=fill
        )
        if labels:
            draw_label(
                draw, obj['bbox'][:2], obj['label'], obj['prob'], color,
                scale=scale
            )

    return image
