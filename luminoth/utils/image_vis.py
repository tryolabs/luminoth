import io
import numpy as np
import os
import logging
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
import tensorflow as tf

from .bbox_overlap import bbox_overlap
from .bbox_transform import decode
from base64 import b64encode
from sys import stdout

# flake8: noqa

font = ImageFont.load_default()
logger = logging.getLogger('luminoth-vis')


summaries_fn = {
    'fasterrcnn': {
        'train': {
            'rpn': {
                'draw_top_nms_proposals': None,
                'draw_gt_boxes': None,
            },
            'rcnn': {
                'draw_object_prediction': None
            }
        },
        'eval': {
            'rpn': {
                'draw_top_nms_proposals': None,
                'draw_gt_boxes': None,
            },
            'rcnn': {
                'draw_object_prediction': None
            }
        },
        'debug': {
            'rpn': {
                'draw_anchors': [
                    None, {'anchor_num': 0}
                ],
                'draw_anchor_centers': None,
                'draw_anchor_batch': None,
                'draw_positive_anchors': None,
                'draw_top_proposals': [
                    None, {'top_k': False}, {'max_display': 50},
                ],
                'draw_top_nms_proposals': [
                    None, {'min_score': 0.9},
                    {'min_score': 0.75}, {'min_score': 0.05}
                ],
                'draw_batch_proposals': [
                    {'display': 'anchor'}, {'display': 'proposal'},
                    {'display': 'proposal', 'draw_all': False},
                    {'display': 'proposal', 'top_k': 10, 'draw_all': False},
                    {'display': 'proposal', 'top_k': 20, 'draw_all': False},
                    {'display': 'anchor', 'top_k': 10, 'draw_all': False},
                    {'display': 'anchor', 'top_k': 20, 'draw_all': False},
                ],
                'draw_rpn_cls_loss': [
                    {'foreground': True, 'topn': 10, 'worst': True},
                    {'foreground': True, 'topn': 10, 'worst': False},
                    {'foreground': False, 'topn': 10, 'worst': True},
                    {'foreground': False, 'topn': 10, 'worst': False},
                    {'foreground': True, 'topn': 20, 'worst': True},
                    {'foreground': True, 'topn': 20, 'worst': False},
                    {'foreground': False, 'topn': 20, 'worst': True},
                    {'foreground': False, 'topn': 20, 'worst': False},
                ],
                'draw_rpn_bbox_targets': None,
                'draw_rpn_bbox_pred': [
                    {'top_k': 1}, {'top_k': 5}, {'top_k': 10}, {'top_k': 20},
                    {'top_k': 40}, {'top_k': 80}
                ],
                'draw_rpn_bbox_pred_with_target': [
                    {'worst': True}, {'worst': False}
                ],
                'draw_gt_boxes': None,
                'draw_correct_rpn_proposals_anchors': None,
                'draw_rpn_pred_combined_loss': [
                    {'top_k': 1}, {'top_k': 5}, {'top_k': 10}, {'top_k': 20},
                    {'top_k': 50}
                ],
            },
            'rcnn': {
                'draw_rcnn_cls_batch': None,
                'draw_rcnn_input_proposals': None,
                'draw_rcnn_cls_batch_errors': [
                    {'worst': True}, {'worst': False}],
                'draw_object_prediction': None
            }
        },
    }
}


def get_image_summaries(summaries_fn, pred_dict, image,
                        gt_bboxes=None, extra_tag=None):
    summaries = []
    if gt_bboxes is not None:
        pred_dict['gt_bboxes'] = gt_bboxes

    for fn_name, arguments in summaries_fn.items():
        if not arguments:
            arguments = [{}]

        for argument in arguments:
            if not argument:
                argument = {}
                tag = fn_name
            else:
                tag = os.path.join(fn_name, ','.join(
                    '{}={}'.format(k, v) for k, v in argument.items()
                ))
            if extra_tag:
                tag = '{}/{}'.format(tag, extra_tag)
            try:
                summary = image_to_summary(
                    globals()[fn_name](
                        pred_dict, image,
                        **argument), tag)
                summaries.append(summary)
            except KeyError as err:
                tf.logging.warning(
                    'Function {} failed with KeyError. Key value: {}'.format(
                        fn_name, err))
    return summaries


def image_vis_summaries(pred_dict, with_rcnn=True, extra_tag=None,
                        image_visualization_mode=None, image=None,
                        gt_bboxes=None):
    summaries = []
    summaries.extend(
        get_image_summaries(
            summaries_fn[
                'fasterrcnn'][image_visualization_mode]['rpn'],
            pred_dict, image, gt_bboxes,
            extra_tag=extra_tag
        )
    )
    if with_rcnn and summaries_fn[
            'fasterrcnn'][image_visualization_mode].get('rcnn'):
        summaries.extend(
            get_image_summaries(
                summaries_fn[
                    'fasterrcnn'][image_visualization_mode]['rcnn'],
                pred_dict, image, gt_bboxes, extra_tag=extra_tag
            )
        )

    return summaries


def image_to_summary(image_pil, tag):
    summary = tf.Summary(value=[
        tf.Summary.Value(tag=tag, image=tf.Summary.Image(
            encoded_image_string=imagepil_to_str(image_pil)))
    ])
    return summary


def imagepil_to_str(image_pil):
    output = io.BytesIO()
    image_pil.save(output, format='PNG')
    png_string = output.getvalue()
    output.close()
    return png_string


def imgcat(data, width='auto', height='auto', preserveAspectRatio=False,
           inline=True, filename=''):
    """
    The width and height are given as a number followed by a unit, or the
            word "auto".
        N: N character cells.
        Npx: N pixels.
        N%: N percent of the session's width or height.
        auto: The image's inherent size will be used to determine an
            appropriate dimension.
    """
    buf = bytes()
    enc = 'utf-8'

    is_tmux = os.environ['TERM'].startswith('screen')

    # OSC
    buf += b'\033'
    if is_tmux:
        buf += b'Ptmux;\033\033'
    buf += b']'

    buf += b'1337;File='

    if filename:
        buf += b'name='
        buf += b64encode(filename.encode(enc))

    buf += b';size=%d' % len(data)
    buf += b';inline=%d' % int(inline)
    buf += b';width=%s' % width.encode(enc)
    buf += b';height=%s' % height.encode(enc)
    buf += b';preserveAspectRatio=%d' % int(preserveAspectRatio)
    buf += b':'
    buf += b64encode(data)

    # ST
    buf += b'\a'
    if is_tmux:
        buf += b'\033\\'

    buf += b'\n'

    stdout.buffer.write(buf)
    stdout.flush()


def imgcat_pil(image_pil):
    image_bytes = io.BytesIO()
    image_pil.save(image_bytes, format='PNG')
    imgcat(image_bytes.getvalue())


def get_image_draw(image):
    image_pil = Image.fromarray(
        np.uint8(np.squeeze(image))).convert('RGB')
    draw = ImageDraw.Draw(image_pil, 'RGBA')
    return image_pil, draw


def draw_positive_anchors(pred_dict, image):
    """
    Draws positive anchors used as "correct" in RPN
    """
    anchors = pred_dict['all_anchors']
    correct_labels = pred_dict['rpn_prediction']['rpn_cls_target']
    correct_labels = np.squeeze(correct_labels.reshape(anchors.shape[0], 1))
    positive_indices = np.nonzero(correct_labels > 0)[0]
    positive_anchors = anchors[positive_indices]
    correct_labels = correct_labels[positive_indices]
    gt_bboxes = pred_dict['gt_bboxes']
    max_overlap = pred_dict['rpn_prediction']['rpn_max_overlap']
    max_overlap = np.squeeze(max_overlap.reshape(anchors.shape[0], 1))
    overlap_iou = max_overlap[positive_indices]

    image_pil, draw = get_image_draw(image)

    logger.debug(
        'We have {} positive_anchors'.format(positive_anchors.shape[0]))
    logger.debug('GT boxes: {}'.format(gt_bboxes))

    for label, positive_anchor in zip(list(overlap_iou), positive_anchors):
        draw.rectangle(
            list(positive_anchor), fill=(255, 0, 0, 40),
            outline=(0, 255, 0, 100))
        x, y = positive_anchor[:2]
        x = max(x, 0)
        y = max(y, 0)
        draw.text(
            tuple([x, y]), text=str(label), font=font,
            fill=(0, 255, 0, 255))

    for gt_box in gt_bboxes:
        draw.rectangle(
            list(gt_box[:4]), fill=(0, 0, 255, 60),
            outline=(0, 0, 255, 150))

    return image_pil


def draw_gt_boxes(pred_dict, image):
    """
    Draws GT boxes.
    """

    image_pil, draw = get_image_draw(image)
    gt_bboxes = pred_dict['gt_bboxes']
    for gt_box in gt_bboxes:
        draw.rectangle(
            list(gt_box[:4]),
            fill=(0, 0, 255, 60),
            outline=(0, 0, 255, 150)
        )

    return image_pil


def draw_anchor_centers(pred_dict, image):
    anchors = pred_dict['all_anchors']
    x_min = anchors[:, 0]
    y_min = anchors[:, 1]
    x_max = anchors[:, 2]
    y_max = anchors[:, 3]

    center_x = x_min + (x_max - x_min) / 2.
    center_y = y_min + (y_max - y_min) / 2.

    image_pil, draw = get_image_draw(image)

    for x, y in zip(center_x, center_y):
        draw.rectangle(
            [x - 1, y - 1, x + 1, y + 1],
            fill=(255, 0, 0, 150), outline=(0, 255, 0, 200)
        )

    return image_pil


def draw_anchors(pred_dict, image, anchor_num=None):
    """
    Draws positive anchors used as "correct" in RPN
    """
    anchors = pred_dict['all_anchors']
    x_min = anchors[:, 0]
    y_min = anchors[:, 1]
    x_max = anchors[:, 2]
    y_max = anchors[:, 3]

    height = pred_dict['image_shape'][0]
    width = pred_dict['image_shape'][1]

    areas = np.unique(np.round((x_max - x_min) * (y_max - y_min)))

    inside_filter = np.logical_and.reduce((
        (x_min >= 0),
        (y_min >= 0),
        (x_max < width),
        (y_max < height)
    ))

    x_negative = x_min < 0
    y_negative = y_min < 0
    x_outof = x_max >= width
    y_outof = y_max >= height

    if anchor_num is None:
        logger.debug('{} unique areas: {}'.format(len(areas), areas))
        logger.debug('{:.2f}% valid anchors'.format(
            100.0 * np.count_nonzero(inside_filter) /
            inside_filter.shape[0]
        ))
        logger.debug('''
{} anchors with X_min negative.
{} anchors with Y_min negative.
{} anchors with X_max above limit.
{} anchors with Y_max above limit.
        '''.format(
            np.count_nonzero(x_negative),
            np.count_nonzero(y_negative),
            np.count_nonzero(x_outof),
            np.count_nonzero(y_outof),
        ))

    moved_anchors = anchors.copy()
    min_x = -x_min.min()
    min_y = -y_min.min()

    moved_anchors += [[min_x, min_y, min_x, min_y]]

    max_x = int(moved_anchors[:, 2].max())
    max_y = int(moved_anchors[:, 3].max())

    image_pil, _ = get_image_draw(image)
    back = Image.new('RGB', [max_x, max_y], 'white')
    back.paste(image_pil, [int(min_x), int(min_y)])

    draw = ImageDraw.Draw(back, 'RGBA')

    anchor_id_to_draw = anchor_num
    draw_every = pred_dict['anchor_reference'].shape[0]
    first = True
    for anchor_id, anchor in enumerate(moved_anchors):
        if anchor_num is not None:
            if anchor_id != anchor_id_to_draw:
                continue
            else:
                anchor_id_to_draw += draw_every

        if first and anchor_num is not None:
            draw.rectangle(
                list(anchor), fill=(255, 0, 0, 40), outline=(0, 255, 0, 120)
            )
            first = False
        elif anchor_num is None:
            draw.rectangle(
                list(anchor), fill=(255, 0, 0, 1), outline=(0, 255, 0, 2)
            )
        else:
            draw.rectangle(
                list(anchor), fill=(255, 0, 0, 2), outline=(0, 255, 0, 4)
            )

    draw.text(
        tuple([min_x, min_y - 10]),
        text='{}w x {}h'.format(width, height),
        font=font, fill=(0, 0, 0, 160)
    )

    return back


def draw_anchor_batch(pred_dict, image):
    """
    Draw anchors used in the batch for RPN.
    """
    anchors = pred_dict['all_anchors']
    targets = pred_dict['rpn_prediction']['rpn_cls_target']

    in_batch_idx = targets >= 0
    anchors = anchors[in_batch_idx]
    targets = targets[in_batch_idx]

    image_pil, draw = get_image_draw(image)

    for anchor, target in zip(anchors, targets):
        if target == 1:
            draw.rectangle(
                list(anchor), fill=(20, 200, 10, 15),
                outline=(20, 200, 10, 30))
        else:
            draw.rectangle(
                list(anchor), fill=(200, 10, 170, 10),
                outline=(200, 10, 170, 30))

    return image_pil


def draw_bbox(image, bbox):
    """
    bbox: x1,y1,x2,y2
    image: h,w,rgb
    """
    image_pil = Image.fromarray(np.uint8(image)).convert('RGB')

    draw = ImageDraw.Draw(image_pil, 'RGBA')
    draw.rectangle(bbox, fill=(255, 0, 0, 60), outline=(0, 255, 0, 200))

    return image_pil


def draw_top_proposals(pred_dict, image, min_score=0.8, max_display=20,
                       top_k=True, used_in_batch=False):
    tf.logging.debug(
        'Top proposals (blue = matches target in batch, '
        'green = matches background in batch, red = ignored in batch)')
    proposal_prediction = pred_dict['rpn_prediction']['proposal_prediction']
    if top_k:
        scores = proposal_prediction['sorted_top_scores']
        proposals = proposal_prediction['sorted_top_proposals']
    else:
        scores = proposal_prediction['scores']
        proposals = proposal_prediction['proposals']

    min_score_idx = scores > min_score
    scores = scores[min_score_idx]
    proposals = proposals[min_score_idx]

    top_scores_idx = np.argsort(scores)[::-1][:max_display]
    scores = scores[top_scores_idx]
    proposals = proposals[top_scores_idx]

    image_pil, draw = get_image_draw(image)

    for proposal, score in zip(proposals, scores):
        bbox = list(proposal)
        if (bbox[2] - bbox[0] <= 0) or (bbox[3] - bbox[1] <= 0):
            logger.debug(
                'Ignoring top proposal without positive area: '
                '{}, score: {}'.format(proposal, score))
            continue

        draw.rectangle(
            list(bbox), fill=(0, 255, 0, 20),
            outline=(0, 255, 0, 80))
        x, y = bbox[:2]
        x = max(x, 0)
        y = max(y, 0)

        draw.text(
            tuple([x, y]), text=str(score), font=font,
            fill=(30, 30, 30, 80))

    return image_pil


def draw_batch_proposals(pred_dict, image, display='proposal', top_k=None,
                         draw_all=True):
    tf.logging.debug(
        'Batch proposals (background or foreground) '
        '(score is classification, blue = foreground, '
        'red = background, green = GT)')
    tf.logging.debug(
        'This only displays the images on the batch (256). '
        'The number displayed is the classification score '
        '(green is > 0.5, red <= 0.5)')
    tf.logging.debug(
        '{} are displayed'.format(
            'Anchors' if display == 'anchor' else 'Final proposals'))
    scores = pred_dict['rpn_prediction']['rpn_cls_prob']
    scores = scores[:, 1]
    bbox_pred = pred_dict['rpn_prediction']['rpn_bbox_pred']
    targets = pred_dict['rpn_prediction']['rpn_cls_target']
    max_overlaps = pred_dict['rpn_prediction']['rpn_max_overlap']
    all_anchors = pred_dict['all_anchors']
    gt_bboxes = pred_dict['gt_bboxes']

    batch_idx = targets >= 0
    scores = scores[batch_idx]
    bbox_pred = bbox_pred[batch_idx]
    max_overlaps = max_overlaps[batch_idx]
    all_anchors = all_anchors[batch_idx]
    targets = targets[batch_idx]

    if top_k:
        top_scores_idx = np.argsort(scores)[::-1][:top_k]
        scores = scores[top_scores_idx]
        bbox_pred = bbox_pred[top_scores_idx]
        max_overlaps = max_overlaps[top_scores_idx]
        all_anchors = all_anchors[top_scores_idx]
        targets = targets[top_scores_idx]

    if not draw_all:
        # Only batch foreground.
        foreground_idx = targets > 0
        scores = scores[foreground_idx]
        bbox_pred = bbox_pred[foreground_idx]
        max_overlaps = max_overlaps[foreground_idx]
        all_anchors = all_anchors[foreground_idx]
        targets = targets[foreground_idx]

    bboxes = decode(all_anchors, bbox_pred)

    image_pil, draw = get_image_draw(image)

    for score, proposal, target, max_overlap, anchor in zip(
            scores, bboxes, targets, max_overlaps, all_anchors):

        if ((proposal[2] - proposal[0] <= 0) or
                (proposal[3] - proposal[1] <= 0)):
            logger.debug(
                'Ignoring proposal for target {} '
                'because of negative area => {}'.format(
                    target, proposal))
            continue

        if target == 1:
            fill = (0, 0, 255, 30)
            if score > 0.8:
                font_fill = (0, 0, 255, 160)
            else:
                font_fill = (0, 255, 255, 180)
        else:
            fill = (255, 0, 0, 5)
            if score > 0.8:
                font_fill = (255, 0, 255, 160)
            else:
                font_fill = (255, 0, 0, 100)

        if score > 0.5:
            outline_fill = (0, 0, 255, 50)
        else:
            outline_fill = (255, 0, 0, 50)

        if np.abs(score - 1.) < 0.05:
            font_txt = '1'
        else:
            font_txt = '{:.2f}'.format(score)[1:]

        if display == 'anchor':
            box = list(anchor)
        else:
            box = list(proposal)

        draw.rectangle(box, fill=fill, outline=outline_fill)
        x, y = box[:2]
        x = max(x, 0)
        y = max(y, 0)

        score = float(score)
        draw.text(tuple([x, y]), text=font_txt, font=font, fill=font_fill)

    for gt_box in gt_bboxes:
        box = list(gt_box[:4])
        draw.rectangle(box, fill=(0, 255, 0, 60), outline=(0, 255, 0, 70))

    return image_pil


def draw_top_nms_proposals(pred_dict, image, min_score=0.8, draw_gt=False):
    logger.debug('Top NMS proposals (min_score = {})'.format(min_score))
    scores = pred_dict['rpn_prediction']['scores']
    proposals = pred_dict['rpn_prediction']['proposals']
    top_scores_mask = scores > min_score
    scores = scores[top_scores_mask]
    proposals = proposals[top_scores_mask]

    sorted_idx = scores.argsort()[::-1]
    scores = scores[sorted_idx]
    proposals = proposals[sorted_idx]

    image_pil, draw = get_image_draw(image)

    fill_alpha = 70

    for topn, (score, proposal) in enumerate(zip(scores, proposals)):
        bbox = list(proposal)
        if (bbox[2] - bbox[0] <= 0) or (bbox[3] - bbox[1] <= 0):
            logger.debug('Proposal has negative area: {}'.format(bbox))
            continue

        draw.rectangle(
            bbox, fill=(0, 255, 0, fill_alpha), outline=(0, 255, 0, 50))

        if np.abs(score - 1.0) <= 0.01:
            font_txt = '1'
        else:
            font_txt = '{:.2f}'.format(score)[1:]

        draw.text(
            tuple([bbox[0], bbox[1]]), text=font_txt,
            font=font, fill=(0, 255, 0, 150))

        fill_alpha -= 5

    if draw_gt:
        gt_bboxes = pred_dict['gt_bboxes']
        for gt_box in gt_bboxes:
            draw.rectangle(
                list(gt_box[:4]), fill=(0, 0, 255, 60),
                outline=(0, 0, 255, 150))

    return image_pil


def draw_rpn_cls_loss(pred_dict, image, foreground=True, topn=10,
                      worst=True):
    """
    For each bounding box labeled object. We want to display the softmax score.

    We display the anchors, and not the adjusted bounding boxes.
    """
    loss = pred_dict['rpn_prediction']['cross_entropy_per_anchor']
    type_str = 'foreground' if foreground else 'background'
    logger.debug(
        'RPN classification loss (anchors, with the softmax score) '
        '(mean softmax loss (all): {})'.format(loss.mean()))
    logger.debug('Showing {} only'.format(type_str))
    logger.debug('{} {} performers'.format('Worst' if worst else 'Best', topn))
    prob = pred_dict['rpn_prediction']['rpn_cls_prob']
    prob = prob.reshape([-1, 2])[:, 1]
    target = pred_dict['rpn_prediction']['rpn_cls_target']
    anchors = pred_dict['all_anchors']
    gt_bboxes = pred_dict['gt_bboxes']

    non_ignored_indices = target >= 0
    target = target[non_ignored_indices]
    prob = prob[non_ignored_indices]
    anchors = anchors[non_ignored_indices]

    # Get anchors with positive label.
    if foreground:
        positive_indices = np.nonzero(target > 0)[0]
    else:
        positive_indices = np.nonzero(target == 0)[0]

    loss = loss[positive_indices]
    prob = prob[positive_indices]
    anchors = anchors[positive_indices]

    logger.debug('Mean loss for {}: {}'.format(type_str, loss.mean()))

    sorted_idx = loss.argsort()
    if worst:
        sorted_idx = sorted_idx[::-1]

    sorted_idx = sorted_idx[:topn]

    loss = loss[sorted_idx]
    prob = prob[sorted_idx]
    anchors = anchors[sorted_idx]

    logger.debug(
        'Mean loss for displayed {}: {}'.format(type_str, loss.mean()))

    image_pil, draw = get_image_draw(image)

    for anchor_prob, anchor, anchor_loss in zip(prob, anchors, loss):
        anchor = list(anchor)
        draw.rectangle(anchor, fill=(0, 255, 0, 20), outline=(0, 255, 0, 100))
        draw.text(
            tuple([anchor[0], anchor[1]]), text='{:.2f}'.format(anchor_loss),
            font=font, fill=(0, 0, 0, 255))

    for gt_box in gt_bboxes:
        draw.rectangle(
            list(gt_box[:4]), fill=(0, 0, 255, 60), outline=(0, 0, 255, 150))

    return image_pil


def draw_rpn_pred_combined_loss(pred_dict, image, top_k=10):
    target = pred_dict['rpn_prediction']['rpn_cls_target']
    in_target = target >= 0
    bbox_pred = pred_dict['rpn_prediction']['rpn_bbox_pred'][in_target]
    all_anchors = pred_dict['all_anchors'][in_target]
    target = target[in_target]

    in_foreground = target > 0

    ce_loss = pred_dict['rpn_prediction'][
        'cross_entropy_per_anchor'][in_foreground]

    bbox_pred = bbox_pred[in_foreground]
    all_anchors = all_anchors[in_foreground]

    reg_loss = pred_dict['rpn_prediction']['reg_loss_per_anchor']

    combined_loss = ce_loss + reg_loss

    bbox_final = decode(all_anchors, bbox_pred)

    image_pil, draw = get_image_draw(image)

    for bbox, loss in zip(bbox_final, combined_loss):
        bbox = list(bbox)
        draw.rectangle(bbox, fill=(30, 0, 240, 20), outline=(30, 0, 240, 100))
        draw.text(
            tuple([bbox[0], bbox[1]]), text='{:.2f}'.format(loss),
            font=font, fill=(0, 0, 0, 255))

    return image_pil


def draw_rpn_bbox_pred(pred_dict, image, top_k=5):
    """
    For each bounding box labeled object. We want to display bbox_reg_error.

    We display the final bounding box and the anchor. Drawing lines between the
    corners.
    """
    logger.debug(
        'RPN regression loss (bbox to original anchors, '
        'with the smoothL1Loss)')
    target = pred_dict['rpn_prediction']['rpn_cls_target']
    bbox_pred = pred_dict['rpn_prediction']['rpn_bbox_pred']
    all_anchors = pred_dict['all_anchors']
    # Get anchors with positive label.
    positive_indices = target > 0

    bbox_pred = bbox_pred[positive_indices]
    all_anchors = all_anchors[positive_indices]

    loss_per_anchor = pred_dict['rpn_prediction']['reg_loss_per_anchor']

    top_losses_idx = np.argsort(loss_per_anchor)[::-1][:top_k]

    loss_per_anchor = loss_per_anchor[top_losses_idx]
    bbox_pred = bbox_pred[top_losses_idx]
    all_anchors = all_anchors[top_losses_idx]

    bbox_final = decode(all_anchors, bbox_pred)

    image_pil, draw = get_image_draw(image)

    for anchor, bbox, loss in zip(all_anchors, bbox_final, loss_per_anchor):
        anchor = list(anchor)
        bbox = list(bbox)
        draw.rectangle(anchor, fill=(0, 255, 0, 20), outline=(0, 255, 0, 100))
        draw.rectangle(
            bbox, fill=(255, 0, 255, 20), outline=(255, 0, 255, 100))
        draw.text(
            tuple([anchor[0], anchor[1]]), text='{:.2f}'.format(loss),
            font=font, fill=(0, 0, 0, 255))
        draw.line(
            [(anchor[0], anchor[1]), (bbox[0], bbox[1])],
            fill=(0, 0, 0, 170), width=1)
        draw.line(
            [(anchor[2], anchor[1]), (bbox[2], bbox[1])],
            fill=(0, 0, 0, 170), width=1)
        draw.line(
            [(anchor[2], anchor[3]), (bbox[2], bbox[3])],
            fill=(0, 0, 0, 170), width=1)
        draw.line(
            [(anchor[0], anchor[3]), (bbox[0], bbox[3])],
            fill=(0, 0, 0, 170), width=1)

    return image_pil


def draw_rpn_bbox_targets(pred_dict, image):
    target = pred_dict['rpn_prediction']['rpn_cls_target']
    bbox_target = pred_dict['rpn_prediction']['rpn_bbox_target']
    all_anchors = pred_dict['all_anchors']

    batch_foreground_idxs = target == 1
    bbox_target = bbox_target[batch_foreground_idxs]
    all_anchors = all_anchors[batch_foreground_idxs]

    gt_boxes = decode(all_anchors, bbox_target)

    image_pil, draw = get_image_draw(image)

    for gt_box in gt_boxes:
        draw.rectangle(
            list(gt_box), fill=(30, 0, 240, 20), outline=(30, 0, 240, 100))

    return image_pil


def draw_rpn_bbox_pred_with_target(pred_dict, image, worst=True):
    if worst:
        draw_desc = 'worst'
    else:
        draw_desc = 'best'

    logger.debug(
        'Display prediction vs original for '
        '{} performer or batch.'.format(draw_desc))
    logger.debug(
        'green = anchor, magenta = prediction, '
        'red = anchor * target (should be GT)')
    target = pred_dict['rpn_prediction']['rpn_cls_target']
    target = target.reshape([-1, 1])
    # Get anchors with positive label.
    positive_indices = np.nonzero(np.squeeze(target) > 0)[0]
    random_indices = np.random.choice(np.arange(len(positive_indices)), 5)

    loss_per_anchor = pred_dict['rpn_prediction']['reg_loss_per_anchor']

    # Get only n random to avoid overloading image.
    positive_indices = positive_indices[random_indices]
    loss_per_anchor = loss_per_anchor[random_indices]
    target = target[positive_indices]

    bbox_pred = pred_dict['rpn_prediction']['rpn_bbox_pred']
    bbox_pred = bbox_pred.reshape([-1, 4])
    bbox_pred = bbox_pred[positive_indices]

    bbox_target = pred_dict['rpn_prediction']['rpn_bbox_target']
    bbox_target = bbox_target.reshape([-1, 4])
    bbox_target = bbox_target[positive_indices]

    all_anchors = pred_dict['all_anchors']
    all_anchors = all_anchors[positive_indices]

    if worst:
        loss_idx = loss_per_anchor.argmax()
    else:
        loss_idx = loss_per_anchor.argmin()

    loss = loss_per_anchor[loss_idx]
    anchor = all_anchors[loss_idx]
    bbox_pred = bbox_pred[loss_idx]
    bbox_target = bbox_target[loss_idx]

    bbox = decode(np.array([anchor]), np.array([bbox_pred]))[0]
    gt_box = decode(np.array([anchor]), np.array([bbox_target]))[0]

    image_pil, draw = get_image_draw(image)

    anchor = list(anchor)
    bbox = list(bbox)
    gt_box = list(gt_box)
    draw.rectangle(anchor, fill=(0, 255, 0, 20), outline=(0, 255, 0, 100))
    draw.rectangle(bbox, fill=(255, 0, 255, 20), outline=(255, 0, 255, 100))
    draw.rectangle(gt_box, fill=(255, 0, 0, 20), outline=(255, 0, 0, 100))

    logger.debug('Loss is {}'.format(loss))
    return image_pil


def draw_rcnn_cls_batch(pred_dict, image, foreground=True, background=True):
    logger.debug(
        'Show the proposals used for training classifier. '
        '(GT labels are -1 from cls targets)')
    logger.debug('blue => GT, green => foreground, red => background')

    proposals = pred_dict['rpn_prediction']['proposals']
    cls_targets = pred_dict['classification_prediction']['target']['cls']
    gt_bboxes = pred_dict['gt_bboxes']

    batch_idx = np.where(cls_targets != -1)[0]
    bboxes = proposals[batch_idx]
    cls_targets = cls_targets[batch_idx]

    image_pil, draw = get_image_draw(image)

    for bbox, cls_target in zip(bboxes, cls_targets):
        bbox = list(bbox.astype(int))
        if cls_target > 0:
            fill = (0, 255, 0, 20)
            outline = (0, 255, 0, 100)
        else:
            fill = (255, 0, 0, 20)
            outline = (255, 0, 0, 100)

        draw.rectangle(bbox, fill=fill, outline=outline)
        draw.text(
            tuple(bbox[:2]), text=str(int(cls_target)), font=font, fill=fill)

    for gt_box in gt_bboxes:
        draw.rectangle(
            list(gt_box[:4]), fill=(0, 0, 255, 20), outline=(0, 0, 255, 100))
        draw.text(
            tuple(gt_box[:2]), text=str(gt_box[4]),
            font=font, fill=(0, 0, 255, 255))

    return image_pil


def draw_rcnn_cls_batch_errors(pred_dict, image, foreground=True,
                               background=True, worst=True, n=10):
    logger.debug(
        'Show the {} classification errors in batch '
        'used for training classifier.'.format('worst' if worst else 'best'))
    logger.debug('blue => GT, green => foreground, red => background')

    proposals = pred_dict['rpn_prediction']['proposals']
    cls_targets = pred_dict['classification_prediction']['target']['cls']
    bbox_offsets_targets = pred_dict[
        'classification_prediction']['target']['bbox_offsets']
    gt_bboxes = pred_dict['gt_bboxes']
    batch_idx = np.where(cls_targets != -1)[0]

    proposals = proposals[batch_idx]
    cls_targets = cls_targets[batch_idx]
    bbox_offsets_targets = bbox_offsets_targets[batch_idx]

    # Cross entropy per proposal already has >= 0 target
    # batches (not ignored proposals)
    cross_entropy_per_proposal = pred_dict[
        'classification_prediction']['_debug'][
        'losses']['cross_entropy_per_proposal']

    if worst:
        selected_idx = cross_entropy_per_proposal.argsort()[::-1][:n]
    else:
        selected_idx = cross_entropy_per_proposal.argsort()[:n]

    cross_entropy_per_proposal = cross_entropy_per_proposal[selected_idx]
    proposals = proposals[selected_idx]
    cls_targets = cls_targets[selected_idx]
    bbox_offsets_targets = bbox_offsets_targets[selected_idx]

    bboxes = decode(proposals, bbox_offsets_targets)

    image_pil, draw = get_image_draw(image)

    for bbox, cls_target, error in zip(
            bboxes, cls_targets, cross_entropy_per_proposal):
        bbox = list(bbox.astype(int))
        if cls_target > 0:
            fill = (0, 255, 0, 20)
            outline = (0, 255, 0, 100)
        else:
            fill = (255, 0, 0, 20)
            outline = (255, 0, 0, 100)

        draw.rectangle(bbox, fill=fill, outline=outline)
        draw.text(
            tuple(bbox[:2]), text='{:.2f}'.format(error), font=font, fill=fill)

    for gt_box in gt_bboxes:
        draw.rectangle(
            list(gt_box[:4]), fill=(0, 0, 255, 20), outline=(0, 0, 255, 100))
        # draw.text(tuple(gt_box[:2]), text=str(gt_box[4]),
        #       font=font, fill=(0, 0, 255, 255))

    return image_pil


def draw_rcnn_reg_batch_errors(pred_dict, image):
    logger.debug(
        'Show errors in batch used for training classifier regressor.')
    logger.debug(
        'blue => GT, green => foreground, '
        'r`regression_loss` - c`classification_loss`.')

    proposals = pred_dict['rpn_prediction']['proposals']
    cls_targets = pred_dict['classification_prediction']['target']['cls']
    bbox_offsets_targets = pred_dict[
        'classification_prediction']['target']['bbox_offsets']
    bbox_offsets = pred_dict['classification_prediction']['bbox_offsets']
    gt_bboxes = pred_dict['gt_bboxes']

    batch_idx = np.where(cls_targets >= 0)[0]

    proposals = proposals[batch_idx]
    cls_targets = cls_targets[batch_idx]
    bbox_offsets_targets = bbox_offsets_targets[batch_idx]
    bbox_offsets = bbox_offsets[batch_idx]
    cross_entropy_per_proposal = pred_dict[
        'classification_prediction']['_debug'][
        'losses']['cross_entropy_per_proposal']

    foreground_batch_idx = np.where(cls_targets > 0)[0]

    proposals = proposals[foreground_batch_idx]
    cls_targets = cls_targets[foreground_batch_idx]
    bbox_offsets_targets = bbox_offsets_targets[foreground_batch_idx]
    bbox_offsets = bbox_offsets[foreground_batch_idx]
    cross_entropy_per_proposal = cross_entropy_per_proposal[
        foreground_batch_idx]
    reg_loss_per_proposal = pred_dict[
        'classification_prediction']['_debug'][
        'losses']['reg_loss_per_proposal']

    cls_targets = cls_targets - 1

    bbox_offsets_idx_pairs = np.stack(
        np.array([
            cls_targets * 4, cls_targets * 4 + 1,
            cls_targets * 4 + 2, cls_targets * 4 + 3]
        ), axis=1)
    bbox_offsets = np.take(bbox_offsets, bbox_offsets_idx_pairs.astype(np.int))

    bboxes = decode(proposals, bbox_offsets)

    image_pil, draw = get_image_draw(image)

    for proposal, bbox, cls_target, reg_error, cls_error in zip(
            proposals, bboxes, cls_targets,
            reg_loss_per_proposal, cross_entropy_per_proposal):
        bbox = list(bbox.astype(int))
        proposal = list(proposal.astype(int))

        if cls_target > 0:
            fill = (0, 255, 0, 20)
            outline = (0, 255, 0, 100)
            proposal_fill = (255, 255, 30, 20)
            proposal_outline = (255, 255, 30, 100)
        else:
            fill = (255, 0, 0, 20)
            outline = (255, 0, 0, 100)
            proposal_fill = (255, 30, 255, 20)
            proposal_outline = (255, 30, 255, 100)

        draw.rectangle(bbox, fill=fill, outline=outline)
        draw.rectangle(proposal, fill=proposal_fill, outline=proposal_outline)
        draw.text(
            tuple(bbox[:2]), text='r{:.3f} - c{:.2f}'.format(
                reg_error, cls_error), font=font, fill=(0, 0, 0, 150))

        draw.line(
            [(proposal[0], proposal[1]), (bbox[0], bbox[1])],
            fill=(0, 0, 0, 170), width=1)
        draw.line(
            [(proposal[2], proposal[1]), (bbox[2], bbox[1])],
            fill=(0, 0, 0, 170), width=1)
        draw.line(
            [(proposal[2], proposal[3]), (bbox[2], bbox[3])],
            fill=(0, 0, 0, 170), width=1)
        draw.line(
            [(proposal[0], proposal[3]), (bbox[0], bbox[3])],
            fill=(0, 0, 0, 170), width=1)

    for gt_box in gt_bboxes:
        draw.rectangle(
            list(gt_box[:4]), fill=(0, 0, 255, 20), outline=(0, 0, 255, 100))

    return image_pil


def recalculate_objects(pred_dict, image):
    proposals = pred_dict['rpn_prediction']['proposals']
    proposals_prob = pred_dict['classification_prediction']['rcnn']['cls_prob']
    proposals_target = proposals_prob.argmax(axis=1) - 1
    bbox_offsets = pred_dict[
        'classification_prediction']['rcnn']['bbox_offsets']

    bbox_offsets = bbox_offsets[proposals_target >= 0]
    proposals = proposals[proposals_target >= 0]
    proposals_target = proposals_target[proposals_target >= 0]

    bbox_offsets_idx_pairs = np.stack(
        np.array([
            proposals_target * 4, proposals_target * 4 + 1,
            proposals_target * 4 + 2, proposals_target * 4 + 3]), axis=1)
    bbox_offsets = np.take(bbox_offsets, bbox_offsets_idx_pairs.astype(np.int))

    bboxes = decode(proposals, bbox_offsets)

    return bboxes, proposals_target


def draw_object_prediction(pred_dict, image, topn=50):
    logger.debug('Display top scored objects with label.')
    objects = pred_dict['classification_prediction']['objects']
    objects_labels = pred_dict['classification_prediction']['labels']
    objects_labels_prob = pred_dict['classification_prediction']['probs']

    if len(objects_labels) == 0:
        logger.debug(
            'No objects detected. Probably all classified as background.')

    image_pil, draw = get_image_draw(image)

    for num_object, (object_, label, prob) in enumerate(
            zip(objects, objects_labels, objects_labels_prob)):
        bbox = list(object_)
        draw.rectangle(bbox, fill=(0, 255, 0, 20), outline=(0, 255, 0, 100))
        draw.text(
            tuple([bbox[0], bbox[1]]), text='{} - {:.2f}'.format(label, prob),
            font=font, fill=(0, 0, 0, 255))

    # bboxes, classes = recalculate_objects(pred_dict)
    # for bbox, label in zip(bboxes, classes):
    #     bbox = list(bbox)
    #     draw.rectangle(bbox, fill=(0, 255, 0, 20), outline=(0, 255, 0, 100))
    #     draw.text(tuple([bbox[0], bbox[1]]), text='{}'.format(label),
    #           font=font, fill=(0, 0, 0, 255))

    return image_pil


def draw_correct_rpn_proposals_anchors(pred_dict, image, top_k=5):
    scores = pred_dict['rpn_prediction']['rpn_cls_prob']
    scores = scores[:, 1]
    bbox_pred = pred_dict['rpn_prediction']['rpn_bbox_pred']
    all_anchors = pred_dict['all_anchors']
    gt_bboxes = pred_dict['gt_bboxes']
    bboxes = decode(all_anchors, bbox_pred)

    gt_overlap = bbox_overlap(bboxes, gt_bboxes[:, :4])
    top_idxs = gt_overlap.max(axis=1).argsort()[::-1][:top_k]
    bboxes = bboxes[top_idxs]
    all_anchors = all_anchors[top_idxs]
    scores = scores[top_idxs]

    image_pil, draw = get_image_draw(image)

    for bbox, anchor, score in zip(bboxes, all_anchors, scores):
        bbox = list(bbox)
        anchor = list(anchor)
        draw.rectangle(bbox, fill=(0, 255, 50, 20), outline=(0, 255, 50, 100))
        draw.rectangle(
            anchor, fill=(0, 50, 255, 20), outline=(0, 50, 255, 100))
        draw.text(
            tuple([bbox[0], bbox[1]]), text='{:.2f}'.format(score)[1:],
            font=font, fill=(0, 0, 0, 255)
        )

    return image_pil


def draw_rpn_correct_proposals(pred_dict, image):
    proposals = pred_dict['rpn_prediction']['proposals']
    gt_bboxes = pred_dict['gt_bboxes']
    gt_boxes = gt_bboxes[:, :4]

    overlaps = bbox_overlap(proposals, gt_boxes)

    top_overlap = overlaps.max(axis=1)

    top_overlap_idx = top_overlap >= 0.95

    proposals = proposals[top_overlap_idx]

    image_pil, draw = get_image_draw(image)

    for proposal in proposals:
        proposal = list(proposal)
        draw.rectangle(
            proposal, fill=(0, 255, 50, 20), outline=(0, 255, 50, 100))


def draw_rcnn_input_proposals(pred_dict, image):
    logger.debug(
        'Display RPN proposals used in training classification. '
        'Top IoU with GT is displayed.')
    proposals = pred_dict['rpn_prediction']['proposals']
    gt_bboxes = pred_dict['gt_bboxes']
    gt_boxes = gt_bboxes[:, :4]

    overlaps = bbox_overlap(proposals, gt_boxes)

    top_overlap = overlaps.max(axis=1)

    top_overlap_idx = top_overlap >= 0.5

    proposals = proposals[top_overlap_idx]
    top_overlap = top_overlap[top_overlap_idx]

    image_pil, draw = get_image_draw(image)

    for proposal, overlap in zip(proposals, top_overlap):
        proposal = list(proposal)
        draw.rectangle(
            proposal, fill=(0, 255, 0, 20), outline=(0, 255, 0, 100))
        draw.text(
            tuple([proposal[0], proposal[1]]),
            text='{:.2f}'.format(overlap)[1:],
            font=font, fill=(0, 0, 0, 255))

    return image_pil
