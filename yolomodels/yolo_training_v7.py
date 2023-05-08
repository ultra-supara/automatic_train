import math
from functools import partial

import tensorflow as tf
from tensorflow.keras import backend as K
from utils.utilsv7.utils_bbox import get_anchors_and_decode


def box_ciou(b1, b2):
    """
    次のように入力します
    ----------
    b1: tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh
    b2: tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh

    次のように返されます
    -------
    ciou: tensor, shape=(batch, feat_w, feat_h, anchor_num, 1)
    """
    #-----------------------------------------------------------#
    #   予測ボックスの左上隅と右下隅を探索します
    #   b1_mins     (batch, feat_w, feat_h, anchor_num, 2)
    #   b1_maxes    (batch, feat_w, feat_h, anchor_num, 2)
    #-----------------------------------------------------------#
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh/2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half
    #-----------------------------------------------------------#
    #   実際のボックスの左上隅と右下隅を探索します
    #   b2_mins     (batch, feat_w, feat_h, anchor_num, 2)
    #   b2_maxes    (batch, feat_w, feat_h, anchor_num, 2)
    #-----------------------------------------------------------#
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh/2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    #-----------------------------------------------------------#
    #   実際のボックスと予測されたボックスの iou をすべて見つけます
    #   iou         (batch, feat_w, feat_h, anchor_num)
    #-----------------------------------------------------------#
    intersect_mins = K.maximum(b1_mins, b2_mins)
    intersect_maxes = K.minimum(b1_maxes, b2_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    union_area = b1_area + b2_area - intersect_area
    iou = intersect_area / K.maximum(union_area, K.epsilon())

    #-----------------------------------------------------------#
    #   computing center gap
    #   center_distance (batch, feat_w, feat_h, anchor_num)
    #-----------------------------------------------------------#
    center_distance = K.sum(K.square(b1_xy - b2_xy), axis=-1)
    enclose_mins = K.minimum(b1_mins, b2_mins)
    enclose_maxes = K.maximum(b1_maxes, b2_maxes)
    enclose_wh = K.maximum(enclose_maxes - enclose_mins, 0.0)
    #-----------------------------------------------------------#
    #   calculate diagonal distance
    #   enclose_diagonal (batch, feat_w, feat_h, anchor_num)
    #-----------------------------------------------------------#
    enclose_diagonal = K.sum(K.square(enclose_wh), axis=-1)
    ciou = iou - 1.0 * (center_distance) / K.maximum(enclose_diagonal ,K.epsilon())

    v = 4 * K.square(tf.math.atan2(b1_wh[..., 0], K.maximum(b1_wh[..., 1], K.epsilon())) - tf.math.atan2(b2_wh[..., 0], K.maximum(b2_wh[..., 1],K.epsilon()))) / (math.pi * math.pi)
    alpha = v /  K.maximum((1.0 - iou + v), K.epsilon())
    ciou = ciou - alpha * v

    ciou = K.expand_dims(ciou, -1)
    return ciou

#---------------------------------------------------#
#   Smooth labeling (v4 , v5, v7)
#---------------------------------------------------#
def _smooth_labels(y_true, label_smoothing):
    num_classes = tf.cast(K.shape(y_true)[-1], dtype=K.floatx())
    label_smoothing = K.constant(label_smoothing, dtype=K.floatx())
    return y_true * (1.0 - label_smoothing) + label_smoothing / num_classes

#---------------------------------------------------#
#   各predict_box とreal_boxのIOUを計算
#---------------------------------------------------#
def box_iou(b1, b2):
    #---------------------------------------------------#
    #   num_anchor,1,4
    #   左上隅と右下隅の座標を計算する
    #---------------------------------------------------#
    b1              = K.expand_dims(b1, -2)
    b1_xy           = b1[..., :2]
    b1_wh           = b1[..., 2:4]
    b1_wh_half      = b1_wh/2.
    b1_mins         = b1_xy - b1_wh_half
    b1_maxes        = b1_xy + b1_wh_half

    #---------------------------------------------------#
    #   1,n,4
    #   左上隅と右下隅の座標を計算します
    #---------------------------------------------------#
    b2              = K.expand_dims(b2, 0)
    b2_xy           = b2[..., :2]
    b2_wh           = b2[..., 2:4]
    b2_wh_half      = b2_wh/2.
    b2_mins         = b2_xy - b2_wh_half
    b2_maxes        = b2_xy + b2_wh_half

    #---------------------------------------------------#
    #   一致領域を計算する
    #---------------------------------------------------#
    intersect_mins  = K.maximum(b1_mins, b2_mins)
    intersect_maxes = K.minimum(b1_maxes, b2_maxes)
    intersect_wh    = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area  = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area         = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area         = b2_wh[..., 0] * b2_wh[..., 1]
    iou             = intersect_area / (b1_area + b2_area - intersect_area)

    return iou

#---------------------------------------------------#
#   lossを計算する関数
#---------------------------------------------------#
def yolo_loss(
    args,
    input_shape,
    anchors,
    anchors_mask,
    num_classes,
    balance         = [0.4, 1.0, 4],
    label_smoothing = 0.01,
    box_ratio       = 0.05,
    obj_ratio       = 1,
    cls_ratio       = 0.5
):
    num_layers = len(anchors_mask)
    #---------------------------------------------------------------------------------------------------#
    #  予測結果を実際のground_truthから分離します。args は [*model_body.output, *y_true] です。
    #   y_true は 3つのfeature_layerを含むリストで、形状は次のとおりです。
    #   (m,20,20,3,85)
    #   (m,40,40,3,85)
    #   (m,80,80,3,85)
    #   yolo_outputs は 3つのfeature_layerを含むリストで、形状は次のとおりです。
    #   (m,20,20,3,85)
    #   (m,40,40,3,85)
    #   (m,80,80,3,85)
    #---------------------------------------------------------------------------------------------------#
    labels          = args[-1]
    y_true          = args[num_layers:-1]
    yolo_outputs    = args[:num_layers]

    #-----------------------------------------------------------#
    #   Get input_shape as 640,640
    #-----------------------------------------------------------#
    input_shape = K.cast(input_shape, K.dtype(y_true[0]))

    loss        = 0
    outputs     = []
    layer_id    = []
    fg_masks    = []
    is_in_boxes_and_centers = []
    #---------------------------------------------------------------------------------------------------#
    #   y_true是一个列表，包含三个特征层，shape分别为(m,20,20,3,85),(m,40,40,3,85),(m,80,80,3,85)。
    #   yolo_outputs是一个列表，包含三个特征层，shape分别为(m,20,20,3,85),(m,40,40,3,85),(m,80,80,3,85)。
    #---------------------------------------------------------------------------------------------------#
    for l in range(num_layers):
        #-----------------------------------------------------------#
        #   将yolo_outputs的特征层输出进行处理、获得四个返回值
        #   grid        (20,20,1,2) 网格坐标
        #   raw_pred    (m,20,20,3,85) 尚未处理的预测结果
        #   pred_xy     (m,20,20,3,2) 解码后的中心坐标
        #   pred_wh     (m,20,20,3,2) 解码后的宽高坐标
        #-----------------------------------------------------------#
        grid, raw_pred, pred_xy, pred_wh = get_anchors_and_decode(yolo_outputs[l],
            anchors[anchors_mask[l]], num_classes, input_shape, calc_loss=True)

        #-----------------------------------------------------------#
        #   pred_box は、decodeされたpredicted_boxの場所
        #   (m,20,20,3,4)
        #-----------------------------------------------------------#
        pred_box = K.concatenate([pred_xy, pred_wh])

        m       = tf.shape(pred_box)[0]
        scale   = tf.cast([[[input_shape[1], input_shape[0], input_shape[1], input_shape[0]]]], tf.float32)
        outputs.append(tf.concat([tf.reshape(pred_box, [m, -1, 4]) * scale, tf.reshape(raw_pred[..., 4:], [m, -1, num_classes + 1])], -1))
        layer_id.append(tf.ones_like(outputs[-1][:, :, 0]) * l)
        fg_masks.append(tf.reshape(y_true[l][..., 0:1], [m, -1]))
        is_in_boxes_and_centers.append(tf.reshape(y_true[l][..., 1:2], [m, -1]))

    outputs     = tf.concat(outputs, 1)
    layer_id    = tf.concat(layer_id, 1)
    fg_masks    = tf.concat(fg_masks, 1)
    is_in_boxes_and_centers = tf.concat(is_in_boxes_and_centers, 1)

    #-----------------------------------------------#
    # [batch, n_anchors_all, 4] 予測枠の座標値
    # [batch, n_anchors_all, 1] 特徴点が対応するオブジェクトを持つかどうか
    # batch, n_anchors_all, n_cls] 特徴点が対応するオブジェクトの種類
    #-----------------------------------------------#
    bbox_preds  = outputs[:, :, :4]
    obj_preds   = outputs[:, :, 4:5]
    cls_preds   = outputs[:, :, 5:]

    #------------------------------------------------------------#
    #   labels                      [batch, max_boxes, 5]
    #   tf.reduce_sum(labels, -1)   [batch, max_boxes]
    #   nlabel                      [batch]
    #------------------------------------------------------------#
    nlabel = tf.reduce_sum(tf.cast(tf.reduce_sum(labels, -1) > 0, K.dtype(outputs)), -1)
    total_num_anchors = tf.shape(outputs)[1]

    num_fg      = 0.0
    loss_obj    = 0.0
    loss_cls    = 0.0
    loss_iou    = 0.0
    def loop_body(b, num_fg, loss_iou, loss_obj, loss_cls):
        # num_gt 1枚の画像に対するreal_boxの数
        num_gt  = tf.cast(nlabel[b], tf.int32)
        #-----------------------------------------------#
        #   gt_bboxes_per_image     [num_gt, 4]
        #   gt_classes              [num_gt]
        #   bboxes_preds_per_image  [n_anchors_all, 4]
        #   obj_preds_per_image     [n_anchors_all, 1]
        #   cls_preds_per_image     [n_anchors_all, num_classes]
        #-----------------------------------------------#
        gt_bboxes_per_image     = labels[b][:num_gt, :4]
        gt_classes              = labels[b][:num_gt,  4]
        bboxes_preds_per_image  = bbox_preds[b]
        obj_preds_per_image     = obj_preds[b]
        cls_preds_per_image     = cls_preds[b]

        def f1():
            num_fg_img  = tf.cast(tf.constant(0), K.dtype(outputs))
            cls_target  = tf.cast(tf.zeros((0, num_classes)), K.dtype(outputs))
            reg_target  = tf.cast(tf.zeros((0, 4)), K.dtype(outputs))
            obj_target  = tf.cast(tf.zeros((total_num_anchors, 1)), K.dtype(outputs))
            fg_mask     = tf.cast(tf.zeros(total_num_anchors), tf.bool)
            return num_fg_img, cls_target, reg_target, obj_target, fg_mask
        def f2():
            fg_mask = tf.cast(fg_masks[b], tf.bool)
            gt_matched_classes, fg_mask, pred_ious_this_matching, matched_gt_inds, num_fg_img = get_assignments(
                fg_mask, gt_bboxes_per_image, gt_classes, bboxes_preds_per_image, obj_preds_per_image, cls_preds_per_image, num_classes, num_gt,
            )
            reg_target  = tf.cast(tf.gather_nd(gt_bboxes_per_image, tf.reshape(matched_gt_inds, [-1, 1])), K.dtype(outputs))
            cls_target  = tf.cast(tf.one_hot(tf.cast(gt_matched_classes, tf.int32), num_classes) * tf.expand_dims(pred_ious_this_matching, -1), K.dtype(outputs))
            obj_target  = tf.cast(tf.expand_dims(fg_mask, -1), K.dtype(outputs))
            return num_fg_img, cls_target, reg_target, obj_target, fg_mask

        num_fg_img, cls_target, reg_target, obj_target, fg_mask = tf.cond(tf.equal(num_gt, 0), f1, f2)
        num_fg      += num_fg_img
        # reg_target = tf.Print(reg_target, [num_fg_img, reg_target, tf.boolean_mask(bboxes_preds_per_image, fg_mask)], summarize=1000)

        _loss_iou   = 1 - box_ciou(reg_target, tf.boolean_mask(bboxes_preds_per_image, fg_mask))
        _loss_obj   = K.binary_crossentropy(_smooth_labels(obj_target, label_smoothing), obj_preds_per_image, from_logits=True)
        _loss_cls   = K.binary_crossentropy(cls_target, tf.boolean_mask(cls_preds_per_image, fg_mask), from_logits=True)
        for layer in range(len(balance)):
            num_pos = tf.maximum(K.sum(tf.cast(tf.logical_and(tf.equal(layer_id[b], layer), fg_mask), tf.float32)), 1)

            loss_iou += K.sum(tf.boolean_mask(_loss_iou, tf.boolean_mask(tf.logical_and(tf.equal(layer_id[b], layer), fg_mask), fg_mask))) * box_ratio / num_pos
            loss_obj += K.mean(tf.boolean_mask(_loss_obj, tf.equal(layer_id[b], layer)) * balance[layer]) * obj_ratio
            loss_cls += K.sum(tf.boolean_mask(_loss_cls, tf.boolean_mask(tf.logical_and(tf.equal(layer_id[b], layer), fg_mask), fg_mask))) * cls_ratio / num_pos / num_classes
        return b + 1, num_fg, loss_iou, loss_obj, loss_cls
    #-----------------------------------------------------------#
    # この場所でループが作られる、画像ごとにループが作られる
    #-----------------------------------------------------------#
    _, num_fg, loss_iou, loss_obj, loss_cls = tf.while_loop(lambda b,*args: b < tf.cast(tf.shape(outputs)[0], tf.int32), loop_body, [0, num_fg, loss_iou, loss_obj, loss_cls])

    num_fg      = tf.cast(tf.maximum(num_fg, 1), K.dtype(outputs))
    loss        = (loss_iou + loss_cls + loss_obj) / tf.cast(tf.shape(outputs)[0], tf.float32)
    # loss = tf.Print(loss, [num_fg, loss_iou / tf.cast(tf.shape(outputs)[0], tf.float32), loss_obj / tf.cast(tf.shape(outputs)[0], tf.float32), loss_cls / tf.cast(tf.shape(outputs)[0], tf.float32) ])
    return loss

"""
The `get_assignments` function calculates the assignment of predicted bounding boxes to ground truth bounding boxes for a single image during the training process of an object detection model.
It computes the IoU (Intersection over Union) and classification loss to determine the best assignments between the predicted and ground truth boxes using the dynamic_k_matching function.

Here's a brief explanation of the function:

1. Filter the predicted bounding box coordinates, objectness scores, and class scores based on the `fg_mask` (foreground mask).
2. Calculate the pair-wise IoUs (`pair_wise_ious`) between ground truth boxes and predicted boxes.
3. Calculate the pair-wise IoU loss (`pair_wise_ious_loss`) by taking the negative logarithm of `pair_wise_ious`.
4. Compute the ground truth class labels in one-hot format (`gt_cls_per_image`) and adjust the predicted class scores (`cls_preds_`) using the sigmoid function and predicted objectness scores.
5. Calculate the pair-wise classification loss (`pair_wise_cls_loss`) using binary cross-entropy between the ground truth class labels and predicted class scores.
6. Compute the `cost` by summing the pair-wise classification loss and 3 times the pair-wise IoU loss.
7. Perform dynamic_k_matching to get the best assignments between predicted boxes and ground truth boxes based on the `cost`.
8. Return the matched ground truth class labels, updated `fg_mask`, predicted IoUs for the matching, matched ground truth indices, and the number of positive samples (feature points).

This function is typically used in the training process of object detection models to calculate the best assignments
    between ground truth and predicted bounding boxes, which in turn helps improve the model's performance.
"""

def get_assignments(fg_mask, gt_bboxes_per_image, gt_classes, bboxes_preds_per_image, obj_preds_per_image, cls_preds_per_image, num_classes, num_gt):
    #-------------------------------------------------------#
    # 実フレーム内の特徴点に対する予測値を得る
    #   fg_mask                 [n_anchors_all]
    #   bboxes_preds_per_image  [fg_mask, 4]
    #   cls_preds_              [fg_mask, num_classes]
    #   obj_preds_              [fg_mask, 1]
    #-------------------------------------------------------#
    bboxes_preds_per_image  = tf.boolean_mask(bboxes_preds_per_image, fg_mask, axis = 0)
    obj_preds_              = tf.boolean_mask(obj_preds_per_image, fg_mask, axis = 0)
    cls_preds_              = tf.boolean_mask(cls_preds_per_image, fg_mask, axis = 0)
    num_in_boxes_anchor     = tf.shape(bboxes_preds_per_image)[0]
    #-------------------------------------------------------#
    #   Calculate the degree of overlap between the true and predicted frames
    #   pair_wise_ious      [num_gt, fg_mask]
    #-------------------------------------------------------#
    # gt_bboxes_per_image = tf.Print(gt_bboxes_per_image, [gt_bboxes_per_image, bboxes_preds_per_image], summarize=1000)
    pair_wise_ious      = box_iou(gt_bboxes_per_image, bboxes_preds_per_image)
    pair_wise_ious_loss = -tf.math.log(pair_wise_ious + 1e-8)
    #-------------------------------------------------------#
    #   Calculate the cross-entropy of the confidence level of the real frame and the predicted frame types
    #   cls_preds_          [num_gt, fg_mask, num_classes]
    #   gt_cls_per_image    [num_gt, fg_mask, num_classes]
    #   pair_wise_cls_loss  [num_gt, fg_mask]
    #-------------------------------------------------------#
    gt_cls_per_image    = tf.tile(tf.expand_dims(tf.one_hot(tf.cast(gt_classes, tf.int32), num_classes), 1), (1, num_in_boxes_anchor, 1))
    cls_preds_          = K.sigmoid(tf.tile(tf.expand_dims(cls_preds_, 0), (num_gt, 1, 1))) *\
                            K.sigmoid(tf.tile(tf.expand_dims(obj_preds_, 0), (num_gt, 1, 1)))

    pair_wise_cls_loss  = tf.reduce_sum(K.binary_crossentropy(gt_cls_per_image, tf.sqrt(cls_preds_)), -1)
    #-------------------------------------------------------#
    # カテゴリーが近いほどclossentropyは低くなる
    # real frameとpredict frameの重なりが多いほど、costは低くなる
    # 特徴点には対応するreal frameが必要であるため、costは低くなる
    #-------------------------------------------------------#
    cost = pair_wise_cls_loss + 3.0 * pair_wise_ious_loss

    gt_matched_classes, fg_mask, pred_ious_this_matching, matched_gt_inds, num_fg = dynamic_k_matching(cost, pair_wise_ious, fg_mask, gt_classes, num_gt)
    return gt_matched_classes, fg_mask, pred_ious_this_matching, matched_gt_inds, num_fg

"""
The `dynamic_k_matching` function is an implementation of the dynamic K-matching algorithm used for object detection models
    , such as YOLO. This algorithm is responsible for matching the ground truth (real) bounding boxes with the predicted bounding boxes generated by the model.
The function calculates the best match for each ground truth box by considering the cost matrix and the intersection over union (IoU) between the predicted and ground truth boxes.

Here's a brief explanation of the function:

1. Initialize `matching_matrix` with zeros, having the same shape as the `cost` matrix.
2. Calculate `n_candidate_k` as the minimum of 20 and the number of columns in `pair_wise_ious`.
3. Calculate the top `n_candidate_k` IoU values (`topk_ious`) for each ground truth box and determine the corresponding `dynamic_ks` value.
4. Loop through each ground truth box, selecting the best matches based on the calculated cost and update the `matching_matrix` accordingly.
5. Calculate the `anchor_matching_gt` tensor by summing the `matching_matrix` along the first axis (ground truth boxes).
6. If a feature point (anchor) is matched with multiple ground truth boxes, update the `matching_matrix` to select the ground truth box with the lowest cost.
7. Update the `fg_mask` based on the selected indices of the positive samples.
8. Obtain the class labels of the matched ground truth boxes and the IoU values for the matching.
9. Return the matched ground truth class labels, updated `fg_mask`, predicted IoUs for the matching, matched ground truth indices, and the number of positive samples (feature points).

This function is typically used in the training process of object detection models to improve the matching
between ground truth and predicted bounding boxes, which in turn enhances the model's performance.
"""

def dynamic_k_matching(cost, pair_wise_ious, fg_mask, gt_classes, num_gt):
    #-------------------------------------------------------#
    #   matching_matrix     [num_gt, fg_mask]
    #   cost                [num_gt, fg_mask]
    #   pair_wise_ious      [num_gt, fg_mask] 每一个真实框和预测框的重合情况
    #   gt_classes          [num_gt]          每一个真实框的种类
    #   fg_mask             [n_anchors_all]
    #-------------------------------------------------------#
    matching_matrix         = tf.zeros_like(cost)

    #------------------------------------------------------------#
    # Pick the n_candidate_k points with the largest iou
    # Get the values of the ten predicted boxes with the largest overlap of the current real boxes
    # The value range of overlap is [0, 1], the value of dynamic_ks is [0, 10]
    # Then sum up and determine how many points should be used for the box prediction
    #   topk_ious           [num_gt, n_candidate_k]
    #   dynamic_ks          [num_gt]
    #   matching_matrix     [num_gt, fg_mask]
    #------------------------------------------------------------#
    n_candidate_k           = tf.minimum(20, tf.shape(pair_wise_ious)[1])
    topk_ious, _            = tf.nn.top_k(pair_wise_ious, n_candidate_k)
    dynamic_ks              = tf.maximum(tf.reduce_sum(topk_ious, 1), 1)
    # dynamic_ks              = tf.Print(dynamic_ks, [topk_ious, dynamic_ks], summarize = 100)

    def loop_body_1(b, matching_matrix):
        #------------------------------------------------------------#
        #   Select the minimum dynamic k points for each real box
        #------------------------------------------------------------#
        _, pos_idx = tf.nn.top_k(-cost[b], k=tf.cast(dynamic_ks[b], tf.int32))
        matching_matrix = tf.concat(
            [matching_matrix[:b], tf.expand_dims(tf.reduce_max(tf.one_hot(pos_idx, tf.shape(cost)[1]), 0), 0), matching_matrix[b+1:]], axis = 0
        )
        # matching_matrix = matching_matrix.write(b, K.cast(tf.reduce_max(tf.one_hot(pos_idx, tf.shape(cost)[1]), 0), K.dtype(cost)))
        return b + 1, matching_matrix
    #-----------------------------------------------------------#
    #   A loop is made in this place, the loop is made for each image
    #-----------------------------------------------------------#
    _, matching_matrix = tf.while_loop(lambda b,*args: b < tf.cast(num_gt, tf.int32), loop_body_1, [0, matching_matrix])

    #------------------------------------------------------------#
    #   anchor_matching_gt  [fg_mask]
    #------------------------------------------------------------#
    anchor_matching_gt = tf.reduce_sum(matching_matrix, 0)
    #------------------------------------------------------------#
    # When a feature point points to multiple real boxes
    # Select the real box with the smallest cost.
    #------------------------------------------------------------#
    biger_one_indice = tf.reshape(tf.where(anchor_matching_gt > 1), [-1])
    def loop_body_2(b, matching_matrix):
        indice_anchor   = tf.cast(biger_one_indice[b], tf.int32)
        indice_gt       = tf.math.argmin(cost[:, indice_anchor])
        matching_matrix = tf.concat(
            [
                matching_matrix[:, :indice_anchor],
                tf.expand_dims(tf.one_hot(indice_gt, tf.cast(num_gt, tf.int32)), 1),
                matching_matrix[:, indice_anchor+1:]
            ], axis = -1
        )
        return b + 1, matching_matrix
    #-----------------------------------------------------------#
    #   A loop is made in this place, the loop is made for each image
    #-----------------------------------------------------------#
    _, matching_matrix = tf.while_loop(lambda b,*args: b < tf.cast(tf.shape(biger_one_indice)[0], tf.int32), loop_body_2, [0, matching_matrix])

    #------------------------------------------------------------#
    # fg_mask_inboxes [fg_mask]
    # num_fg is the number of feature points of positive samples
    #------------------------------------------------------------#
    fg_mask_inboxes = tf.reduce_sum(matching_matrix, 0) > 0.0
    num_fg          = tf.reduce_sum(tf.cast(fg_mask_inboxes, K.dtype(cost)))

    fg_mask_indices         = tf.reshape(tf.where(fg_mask), [-1])
    fg_mask_inboxes_indices = tf.reshape(tf.where(fg_mask_inboxes), [-1, 1])
    fg_mask_select_indices  = tf.gather_nd(fg_mask_indices, fg_mask_inboxes_indices)
    fg_mask                 = tf.cast(tf.reduce_max(tf.one_hot(fg_mask_select_indices, tf.shape(fg_mask)[0]), 0), K.dtype(fg_mask))

    #------------------------------------------------------------#
    # Obtain the item type corresponding to the feature point
    #------------------------------------------------------------#
    matched_gt_inds     = tf.math.argmax(tf.boolean_mask(matching_matrix, fg_mask_inboxes, axis = 1), 0)
    gt_matched_classes  = tf.gather_nd(gt_classes, tf.reshape(matched_gt_inds, [-1, 1]))

    pred_ious_this_matching = tf.boolean_mask(tf.reduce_sum(matching_matrix * pair_wise_ious, 0), fg_mask_inboxes)
    return gt_matched_classes, fg_mask, pred_ious_this_matching, matched_gt_inds, num_fg


"""
The function `get_lr_scheduler` is a learning rate scheduler factory that returns a learning rate scheduling function based on the type of learning rate decay specified.
    It supports two types of learning rate decay: cosine annealing with warm-up and step decay.

Here's a brief explanation of the function:

1. Define `yolox_warm_cos_lr`, a function for cosine annealing with warm-up learning rate scheduling.
2. Define `step_lr`, a function for step decay learning rate scheduling.
3. Check the `lr_decay_type` input to determine which type of scheduler to return.
4. If `lr_decay_type` is "cos", calculate the parameters for the cosine annealing with warm-up scheduler:
    - `warmup_total_iters`: The number of iterations for the warm-up phase.
    - `warmup_lr_start`: The initial learning rate at the start of the warm-up phase.
    - `no_aug_iter`: The number of iterations without augmentation.
    - `func`: A partial function of `yolox_warm_cos_lr` with pre-filled arguments.
5. If `lr_decay_type` is not "cos", calculate the parameters for the step decay scheduler:
    - `decay_rate`: The decay rate for each step.
    - `step_size`: The number of iterations per step.
    - `func`: A partial function of `step_lr` with pre-filled arguments.
6. Return the `func` partial function.

When you call the returned function with the current iteration number as its argument,
    it will return the learning rate for that iteration according to the chosen scheduler.

Please note that the cosine annealing with warm-up scheduler is designed specifically for the YOLOX object detection algorithm
    , and the step decay scheduler is more general-purpose.
"""
def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio = 0.05, warmup_lr_ratio = 0.1, no_aug_iter_ratio = 0.05, step_num = 10):
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2
            ) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                1.0
                + math.cos(
                    math.pi
                    * (iters - warmup_total_iters)
                    / (total_iters - warmup_total_iters - no_aug_iter)
                )
            )
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n       = iters // step_size
        out_lr  = lr * decay_rate ** n
        return out_lr

    if lr_decay_type == "cos":
        warmup_total_iters  = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start     = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter         = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        func = partial(yolox_warm_cos_lr ,lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate  = (min_lr / lr) ** (1 / (step_num - 1))
        step_size   = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)

    return func
