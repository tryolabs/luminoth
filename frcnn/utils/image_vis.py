import io
import numpy as np
import os
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
import tensorflow as tf

from .bbox_transform import bbox_transform_inv
from .bbox import bbox_overlaps
from base64 import b64encode
from sys import stdout


font = ImageFont.load_default()


def imgcat(data, width='auto', height='auto', preserveAspectRatio=False,
           inline=True, filename=''):
    """
    The width and height are given as a number followed by a unit, or the word "auto".
        N: N character cells.
        Npx: N pixels.
        N%: N percent of the session's width or height.
        auto: The image's inherent size will be used to determine an appropriate dimension.
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


def get_image_draw(pred_dict):
    image_pil = Image.fromarray(np.uint8(np.squeeze(pred_dict['image']))).convert('RGB')
    draw = ImageDraw.Draw(image_pil, 'RGBA')
    return image_pil, draw


def draw_positive_anchors(pred_dict):
    """
    Draws positive anchors used as "correct" in RPN
    """
    anchors = pred_dict['all_anchors']
    correct_labels = pred_dict['rpn_prediction']['rpn_cls_target']
    correct_labels = np.squeeze(correct_labels.reshape(anchors.shape[0], 1))
    positive_indices = np.nonzero(correct_labels > 0)[0]
    positive_anchors = anchors[positive_indices]
    correct_labels = correct_labels[positive_indices]

    max_overlap = pred_dict['rpn_prediction']['rpn_max_overlap']
    max_overlap = np.squeeze(max_overlap.reshape(anchors.shape[0], 1))
    overlap_iou = max_overlap[positive_indices]

    gt_boxes = pred_dict['gt_boxes']

    image_pil, draw = get_image_draw(pred_dict)

    print('We have {} positive_anchors'.format(positive_anchors.shape[0]))
    # print('Indices, values and bbox: {}'.format(list(zip(positive_indices, list(overlap_iou), positive_anchors))))
    print('GT boxes: {}'.format(gt_boxes))

    for label, positive_anchor in zip(list(overlap_iou), positive_anchors):
        draw.rectangle(list(positive_anchor), fill=(255, 0, 0, 40), outline=(0, 255, 0, 100))
        x, y = positive_anchor[:2]
        x = max(x, 0)
        y = max(y, 0)
        draw.text(tuple([x, y]), text=str(label), font=font, fill=(0, 255, 0, 255))

    for gt_box in gt_boxes:
        draw.rectangle(list(gt_box[:4]), fill=(0, 0, 255, 60), outline=(0, 0, 255, 150))

    imgcat_pil(image_pil)


def draw_anchors(pred_dict):
    """
    Draws positive anchors used as "correct" in RPN
    """
    print('All anchors')
    anchors = pred_dict['all_anchors']

    image_pil, draw = get_image_draw(pred_dict)

    for anchor_id, anchor in enumerate(anchors):
        draw.rectangle(list(anchor), fill=(255, 0, 0, 2), outline=(0, 255, 0, 6))

    imgcat_pil(image_pil)


def draw_bbox(image, bbox):
    """
    bbox: x1,y1,x2,y2
    image: h,w,rgb
    """
    image_pil = Image.fromarray(np.uint8(image)).convert('RGB')

    draw = ImageDraw.Draw(image_pil, 'RGBA')
    draw.rectangle(bbox, fill=(255, 0, 0, 60), outline=(0, 255, 0, 200))

    imgcat_pil(image_pil)

def draw_top_proposals(pred_dict):
    print('Top proposals (blue = matches target in batch, green = matches background in batch, red = ignored in batch)')
    scores = pred_dict['rpn_prediction']['proposal_prediction']['scores']
    proposals = pred_dict['rpn_prediction']['proposal_prediction']['proposals']
    targets = pred_dict['rpn_prediction']['rpn_cls_target']

    top_proposals_idx = np.where(scores == scores.max())[0]
    scores = scores[top_proposals_idx]
    proposals = proposals[top_proposals_idx]
    targets = targets[top_proposals_idx]

    image_pil, draw = get_image_draw(pred_dict)

    for proposal, target, score in zip(proposals, targets, scores):
        bbox = list(proposal)
        if (bbox[2] - bbox[0] <= 0) or (bbox[3] - bbox[1] <= 0):
            tf.logging.warning('Ignoring top proposal without positive area: {}, score: {}'.format(proposal, score))
            continue

        if target == 1:
            fill = (0, 0, 255, 20)
        elif target == 0:
            fill = (0, 255, 0, 20)
        else:
            fill = (255, 0, 0, 20)

        draw.rectangle(list(bbox), fill=fill, outline=fill)
        x, y = bbox[:2]
        x = max(x, 0)
        y = max(y, 0)

        draw.text(tuple([x, y]), text=str(target), font=font, fill=fill)

    imgcat_pil(image_pil)


def draw_batch_proposals(pred_dict):
    print('Batch proposals (background or foreground) (score is classification, blue = foreground, red = background, green = GT)')
    print('This only displays the images on the batch (256). The number displayed is the classification score (green is > 0.5, red <= 0.5)')
    scores = pred_dict['rpn_prediction']['proposal_prediction']['scores']
    proposals = pred_dict['rpn_prediction']['proposal_prediction']['proposals']
    targets = pred_dict['rpn_prediction']['rpn_cls_target']
    max_overlap = pred_dict['rpn_prediction']['rpn_max_overlap']
    all_anchors = pred_dict['all_anchors']

    image_pil, draw = get_image_draw(pred_dict)

    for score, proposal, target, max_overlap, anchor in zip(scores, proposals, targets, max_overlap, all_anchors):

        if target < 0:
            continue

        if (proposal[2] - proposal[0] <= 0) or (proposal[3] - proposal[1] <= 0):
            tf.logging.warning('Ignoring proposal for target {} because of negative area => {}'.format(target, proposal))
            continue

        if target == 1:
            fill = (0, 0, 255, 10)
            outline_fill = (0, 0, 255, 50)
        else:
            fill = (255, 0, 0, 5)
            outline_fill = (255, 0, 0, 30)

        draw.rectangle(list(proposal), fill=fill, outline=outline_fill)
        x, y = list(proposal)[:2]
        x = max(x, 0)
        y = max(y, 0)

        score = float(score)
        if score > 0.5:
            font_fill = (0, 255, 0, 255)
        else:
            font_fill = (255, 0, 0, 255)

        if np.isclose(score, 1.0) or score > 1.0:
            font_txt = '1'
        else:
            font_txt = '{:.2f}'.format(score)[1:]

        draw.text(tuple([x, y]), text=font_txt, font=font, fill=font_fill)

    gt_boxes = pred_dict['gt_boxes']
    for gt_box in gt_boxes:
        draw.rectangle(list(gt_box[:4]), fill=(0, 255, 0, 60), outline=(0, 255, 0, 150))

    imgcat_pil(image_pil)

def draw_top_nms_proposals(pred_dict, min_score=0.8):
    print('Top NMS proposals')
    scores = pred_dict['rpn_prediction']['scores']
    proposals = pred_dict['rpn_prediction']['proposals']
    # Remove batch id
    proposals = proposals[:,1:]
    top_scores_mask = scores > min_score
    scores = scores[top_scores_mask]
    proposals = proposals[top_scores_mask]
    image_pil, draw = get_image_draw(pred_dict)

    for score, proposal in zip(scores, proposals):
        bbox = list(proposal)
        if (bbox[2] - bbox[0] <= 0) or (bbox[3] - bbox[1] <= 0):
            tf.logging.warning('Proposal has negative area: {}'.format(bbox))
            continue

        draw.rectangle(bbox, fill=(255, 0, 0, 14), outline=(0, 0, 0, 40))
        draw.text(tuple([bbox[0], bbox[1]]), text='{:.2f}'.format(score)[1:], font=font, fill=(0, 255, 0, 150))

    gt_boxes = pred_dict['gt_boxes']
    for gt_box in gt_boxes:
        draw.rectangle(list(gt_box[:4]), fill=(0, 0, 255, 60), outline=(0, 0, 255, 150))

    imgcat_pil(image_pil)


def draw_rpn_cls_loss(pred_dict):
    """
    For each bounding box labeled object. We wan't to display the softmax score.

    We display the anchors, and not the adjusted bounding boxes.
    """
    print('RPN classification loss (anchors, with the softmax score) (mean softmax loss: {})'.format(pred_dict['rpn_prediction']['cross_entropy_per_anchor'].mean()))
    prob = pred_dict['rpn_prediction']['rpn_cls_prob']
    prob = prob.reshape([-1, 2])[:,1]
    loss = pred_dict['rpn_prediction']['cross_entropy_per_anchor']
    target = pred_dict['rpn_prediction']['rpn_cls_target']
    target = target.reshape([-1, 1])
    anchors = pred_dict['all_anchors']

    # Get anchors with positive label.
    positive_indices = np.nonzero(np.squeeze(target) > 0)[0]
    # Get anchors labeled
    labeled_indices = np.nonzero(target[np.nonzero(np.squeeze(target) >= 0)[0]])[0]
    loss = loss[labeled_indices]

    prob = prob[positive_indices]
    anchors = anchors[positive_indices]

    image_pil, draw = get_image_draw(pred_dict)

    for anchor_prob, anchor, anchor_loss in zip(prob, anchors, loss):
        anchor = list(anchor)
        draw.rectangle(anchor, fill=(0, 255, 0, 20), outline=(0, 255, 0, 100))
        draw.text(tuple([anchor[0], anchor[1]]), text='{:.2f}'.format(anchor_loss)[1:], font=font, fill=(0, 0, 0, 255))

    gt_boxes = pred_dict['gt_boxes']
    for gt_box in gt_boxes:
        draw.rectangle(list(gt_box[:4]), fill=(0, 0, 255, 60), outline=(0, 0, 255, 150))

    imgcat_pil(image_pil)


def draw_rpn_bbox_pred(pred_dict, n=5):
    """
    For each bounding box labeled object. We wan't to display the bbox_reg_error

    We display the final bounding box and the anchor. Drawing lines between the
    corners.
    """
    print('RPN regression loss (bbox to original anchors, with the smoothL1Loss)')
    target = pred_dict['rpn_prediction']['rpn_cls_target']
    target = target.reshape([-1, 1])
    # Get anchors with positive label.
    positive_indices = np.nonzero(np.squeeze(target) > 0)[0]
    random_indices = np.random.choice(np.arange(len(positive_indices)), n)

    loss_per_anchor = pred_dict['rpn_prediction']['reg_loss_per_anchor']

    # Get only n random to avoid overloading image.
    positive_indices = positive_indices[random_indices]
    loss_per_anchor = loss_per_anchor[random_indices]
    target = target[positive_indices]

    bbox_pred = pred_dict['rpn_prediction']['rpn_bbox_pred']
    bbox_pred = bbox_pred.reshape([-1, 4])
    bbox_pred = bbox_pred[positive_indices]
    all_anchors = pred_dict['all_anchors']
    all_anchors = all_anchors[positive_indices]

    bbox_final = bbox_transform_inv(all_anchors, bbox_pred)

    image_pil, draw = get_image_draw(pred_dict)

    for anchor, bbox, loss in zip(all_anchors, bbox_final, loss_per_anchor):
        anchor = list(anchor)
        bbox = list(bbox)
        draw.rectangle(anchor, fill=(0, 255, 0, 20), outline=(0, 255, 0, 100))
        draw.rectangle(bbox, fill=(255, 0, 255, 20), outline=(255, 0, 255, 100))
        draw.text(tuple([anchor[0], anchor[1]]), text='{:.2f}'.format(loss), font=font, fill=(0, 0, 0, 255))
        draw.line([(anchor[0], anchor[1]), (bbox[0], bbox[1])], fill=(0,0,0,170), width=1)
        draw.line([(anchor[2], anchor[1]), (bbox[2], bbox[1])], fill=(0,0,0,170), width=1)
        draw.line([(anchor[2], anchor[3]), (bbox[2], bbox[3])], fill=(0,0,0,170), width=1)
        draw.line([(anchor[0], anchor[3]), (bbox[0], bbox[3])], fill=(0,0,0,170), width=1)

    imgcat_pil(image_pil)

def draw_rpn_bbox_pred_with_target(pred_dict, worst=True):
    if worst:
        draw_desc = 'worst'
    else:
        draw_desc = 'best'

    print('Display prediction vs original for {} performer or batch.'.format(draw_desc))
    print('green = anchor, magenta = prediction, red = anchor * target (should be GT)')
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

    bbox = bbox_transform_inv(np.array([anchor]), np.array([bbox_pred]))[0]
    bbox_target = bbox_transform_inv(np.array([anchor]), np.array([bbox_target]))[0]

    image_pil, draw = get_image_draw(pred_dict)

    anchor = list(anchor)
    bbox = list(bbox)
    bbox_target = list(bbox_target)
    draw.rectangle(anchor, fill=(0, 255, 0, 20), outline=(0, 255, 0, 100))
    draw.rectangle(bbox, fill=(255, 0, 255, 20), outline=(255, 0, 255, 100))
    draw.rectangle(bbox_target, fill=(255, 0, 0, 20), outline=(255, 0, 0, 100))

    print('Loss is {}'.format(loss))
    imgcat_pil(image_pil)


def draw_rcnn_cls_batch(pred_dict, foreground=True, background=True):
    print('Show the bboxes used for training classifier. (GT labels are -1 from cls targets)')
    print('blue => GT, green => foreground, red => background')

    proposals = pred_dict['rpn_prediction']['proposals'][:,1:]
    cls_targets = pred_dict['classification_prediction']['cls_target']
    bbox_offsets_targets = pred_dict['classification_prediction']['bbox_offsets_target']

    batch_idx = np.where(cls_targets != -1)[0]

    proposals = proposals[batch_idx]
    cls_targets = cls_targets[batch_idx]
    bbox_offsets_targets = bbox_offsets_targets[batch_idx]

    bboxes = bbox_transform_inv(proposals, bbox_offsets_targets)

    image_pil, draw = get_image_draw(pred_dict)

    for bbox, cls_target in zip(bboxes, cls_targets):
        bbox = list(bbox.astype(int))
        if cls_target > 0:
            fill = (0, 255, 0, 20)
            outline = (0, 255, 0, 100)
        else:
            fill = (255, 0, 0, 20)
            outline = (255, 0, 0, 100)

        draw.rectangle(bbox, fill=fill, outline=outline)
        draw.text(tuple(bbox[:2]), text=str(int(cls_target)), font=font, fill=fill)

    gt_boxes = pred_dict['gt_boxes']
    for gt_box in gt_boxes:
        draw.rectangle(list(gt_box[:4]), fill=(0, 0, 255, 20), outline=(0, 0, 255, 100))
        draw.text(tuple(gt_box[:2]), text=str(gt_box[4]), font=font, fill=(0, 0, 255, 255))

    imgcat_pil(image_pil)


def draw_rcnn_cls_batch_errors(pred_dict, foreground=True, background=True, worst=True, n=10):
    print('Show the {} classification errors in batch used for training classifier.'.format('worst' if worst else 'best'))
    print('blue => GT, green => foreground, red => background')

    proposals = pred_dict['rpn_prediction']['proposals'][:,1:]
    cls_targets = pred_dict['classification_prediction']['cls_target']
    bbox_offsets_targets = pred_dict['classification_prediction']['bbox_offsets_target']

    batch_idx = np.where(cls_targets != -1)[0]

    proposals = proposals[batch_idx]
    cls_targets = cls_targets[batch_idx]
    bbox_offsets_targets = bbox_offsets_targets[batch_idx]

    # Cross entropy per proposal already has >= 0 target batches (not ignored proposals)
    cross_entropy_per_proposal = pred_dict['classification_prediction']['cross_entropy_per_proposal']

    if worst:
        selected_idx = cross_entropy_per_proposal.argsort()[::-1][:n]
    else:
        selected_idx = cross_entropy_per_proposal.argsort()[:n]

    cross_entropy_per_proposal = cross_entropy_per_proposal[selected_idx]
    proposals = proposals[selected_idx]
    cls_targets = cls_targets[selected_idx]
    bbox_offsets_targets = bbox_offsets_targets[selected_idx]

    bboxes = bbox_transform_inv(proposals, bbox_offsets_targets)

    image_pil, draw = get_image_draw(pred_dict)

    for bbox, cls_target, error in zip(bboxes, cls_targets, cross_entropy_per_proposal):
        bbox = list(bbox.astype(int))
        if cls_target > 0:
            fill = (0, 255, 0, 20)
            outline = (0, 255, 0, 100)
        else:
            fill = (255, 0, 0, 20)
            outline = (255, 0, 0, 100)

        draw.rectangle(bbox, fill=fill, outline=outline)
        draw.text(tuple(bbox[:2]), text='{:.2f}'.format(error), font=font, fill=fill)

    gt_boxes = pred_dict['gt_boxes']
    for gt_box in gt_boxes:
        draw.rectangle(list(gt_box[:4]), fill=(0, 0, 255, 20), outline=(0, 0, 255, 100))
        # draw.text(tuple(gt_box[:2]), text=str(gt_box[4]), font=font, fill=(0, 0, 255, 255))

    imgcat_pil(image_pil)


def draw_rcnn_reg_batch_errors(pred_dict, worst=True):
    print('Show the {} regression errors in batch used for training classifier.'.format('worst' if worst else 'best'))
    print('blue => GT, green => foreground, red => background')

    proposals = pred_dict['rpn_prediction']['proposals'][:,1:]
    cls_targets = pred_dict['classification_prediction']['cls_target']
    bbox_offsets_targets = pred_dict['classification_prediction']['bbox_offsets_target']
    bbox_offsets = pred_dict['classification_prediction']['bbox_offsets']

    batch_idx = np.where(cls_targets > 0)[0]

    proposals = proposals[batch_idx]
    cls_targets = cls_targets[batch_idx]
    bbox_offsets_targets = bbox_offsets_targets[batch_idx]
    bbox_offsets = bbox_offsets[batch_idx]
    reg_loss_per_proposal = pred_dict['classification_prediction']['reg_loss_per_proposal']

    cls_targets = cls_targets - 1

    bbox_offsets_idx_pairs = np.stack(np.array([cls_targets * 4, cls_targets * 4 + 1, cls_targets * 4 + 2, cls_targets * 4 + 3]), axis=1)
    bbox_offsets = np.take(bbox_offsets, bbox_offsets_idx_pairs.astype(np.int))

    bboxes = bbox_transform_inv(proposals, bbox_offsets)

    image_pil, draw = get_image_draw(pred_dict)

    for proposal, bbox, cls_target, error in zip(proposals, bboxes, cls_targets, reg_loss_per_proposal):
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
        draw.text(tuple(bbox[:2]), text='{:.3f}'.format(error), font=font, fill=fill)

        draw.line([(proposal[0], proposal[1]), (bbox[0], bbox[1])], fill=(0,0,0,170), width=1)
        draw.line([(proposal[2], proposal[1]), (bbox[2], bbox[1])], fill=(0,0,0,170), width=1)
        draw.line([(proposal[2], proposal[3]), (bbox[2], bbox[3])], fill=(0,0,0,170), width=1)
        draw.line([(proposal[0], proposal[3]), (bbox[0], bbox[3])], fill=(0,0,0,170), width=1)

    gt_boxes = pred_dict['gt_boxes']
    for gt_box in gt_boxes:
        draw.rectangle(list(gt_box[:4]), fill=(0, 0, 255, 20), outline=(0, 0, 255, 100))

    imgcat_pil(image_pil)


def draw_object_prediction(pred_dict, topn=50):
    print('Display top scored objects with label.')
    objects = pred_dict['classification_prediction']['objects']
    objects_labels = pred_dict['classification_prediction']['objects_labels']
    objects_labels_prob = pred_dict['classification_prediction']['objects_labels_prob']

    if len(objects_labels) == 0:
        tf.logging.warning('No objects detected. Probably all classified as background.')

    sorted_idx = objects_labels_prob.argsort()

    objects = objects[sorted_idx]
    objects_labels = objects_labels[sorted_idx]
    objects_labels_prob = objects_labels_prob[sorted_idx]

    image_pil, draw = get_image_draw(pred_dict)

    for num_object, (object_, label, prob) in enumerate(zip(objects, objects_labels, objects_labels_prob)):
        bbox = list(object_)
        draw.rectangle(bbox, fill=(0, 255, 0, 20), outline=(0, 255, 0, 100))
        draw.text(tuple([bbox[0], bbox[1]]), text='{} - {:.2f}'.format(label, prob), font=font, fill=(0, 0, 0, 255))

        if num_object >= topn:
            break

    imgcat_pil(image_pil)


def draw_rcnn_input_proposals(pred_dict):
    proposals = pred_dict['rpn_prediction']['proposals'][:,1:]
    gt_boxes = pred_dict['gt_boxes'][:,:4]

    overlaps = bbox_overlaps(
        np.ascontiguousarray(proposals, dtype=np.float),
        np.ascontiguousarray(gt_boxes, dtype=np.float)
    )

    top_overlap = overlaps.max(axis=1)

    top_overlap_idx = top_overlap >= 0.5

    proposals = proposals[top_overlap_idx]
    top_overlap = top_overlap[top_overlap_idx]

    image_pil, draw = get_image_draw(pred_dict)

    for proposal, overlap in zip(proposals, top_overlap):
        proposal = list(proposal)
        draw.rectangle(proposal, fill=(0, 255, 0, 20), outline=(0, 255, 0, 100))
        draw.text(tuple([proposal[0], proposal[1]]), text='{:.2f}'.format(overlap)[1:], font=font, fill=(0, 0, 0, 255))

    imgcat_pil(image_pil)
