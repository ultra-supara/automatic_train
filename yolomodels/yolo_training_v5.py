"""
This script defines a custom loss function for the YOLO (You Only Look Once) object detection algorithm.
The loss function calculates the CIoU (Complete Intersection over Union) loss, confidence loss, and classification loss.
It uses the CIoU loss to better penalize the differences in the predicted bounding boxes and ground truth bounding boxes.
The confidence loss measures the objectness of the bounding boxes, and the classification loss measures the accuracy of class predictions.

Additionally, the script provides a function `get_lr_scheduler` that returns a learning rate scheduler based on the chosen decay type: cosine annealing or step decay.
The learning rate scheduler adjusts the learning rate during training, typically starting with a higher learning rate and gradually decreasing it.

Here is a brief explanation of the main components of the script:

1. `box_ciou(b1, b2)`: This function calculates the CIoU loss between two sets of bounding boxes.
2. `_smooth_labels(y_true, label_smoothing)`: This function smooths the labels using label smoothing to improve the model's generalization capability.
3. `box_iou(b1, b2)`: This function calculates the Intersection over Union (IoU) between two sets of bounding boxes.
4. `yolo_loss(...)`: This function computes the YOLO loss by combining CIoU loss, confidence loss, and classification loss.
5. `get_lr_scheduler(...)`: This function returns a learning rate scheduler based on the chosen decay type.

This code is written for TensorFlow and Keras, and it is particularly useful for training YOLO-based object detection models.
"""

import math
from functools import partial

import tensorflow as tf
from tensorflow.keras import backend as K
from utils.utilsv5.utils_bbox import get_anchors_and_decode


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
#   Smooth labeling (v4 , v5)
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
    y_true          = args[num_layers:]
    yolo_outputs    = args[:num_layers]

    #-----------------------------------------------------------#
    #   Get input_shape as 416,416
    #-----------------------------------------------------------#
    input_shape = K.cast(input_shape, K.dtype(y_true[0]))

    loss    = 0
    #---------------------------------------------------------------------------------------------------#
    #   y_true是一个列表，包含三个特征层，shape分别为(m,20,20,3,85),(m,40,40,3,85),(m,80,80,3,85)。
    #   yolo_outputs是一个列表，包含三个特征层，shape分别为(m,20,20,3,85),(m,40,40,3,85),(m,80,80,3,85)。
    #---------------------------------------------------------------------------------------------------#
    for l in range(num_layers):
        #-----------------------------------------------------------#
        #   以第一个特征层(m,20,20,3,85)为例子
        #   取出该特征层中存在目标的点的位置。(m,20,20,3,1)
        #-----------------------------------------------------------#
        object_mask = y_true[l][..., 4:5]
        #-----------------------------------------------------------#
        #   Take out the corrsponding type (m,20,20,3,80)
        #-----------------------------------------------------------#
        true_class_probs = y_true[l][..., 5:]
        if label_smoothing:
            true_class_probs = _smooth_labels(true_class_probs, label_smoothing)

        #-----------------------------------------------------------#
        #   yolo_outputs のfeature_layer出力を処理し、4つの戻り値を取得する
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

        #-----------------------------------------------------------#
        #   ciou_lossの計算
        #-----------------------------------------------------------#
        raw_true_box    = y_true[l][...,0:4]
        ciou            = box_ciou(pred_box, raw_true_box)
        ciou_loss       = object_mask * (1 - ciou)

        #------------------------------------------------------------------------------#
        # もしその場所にboxがあったなら、crossentropyを1として信頼度を計算する
        # その場所にboxがなければ、crossentropyを0として信頼度を計算する
        # この処理で無視されるsampleもあり、これらの無視されるサンプルは best_iou < ignore_thresh の条件を満たす
        # 特徴点の点でreal_boxに非常に近く対応する予測は無視される，これらのboxはすでに比較的正確だからだ
        # negative sampleとして扱うには不適切なため、無視されます
        #------------------------------------------------------------------------------#
        tobj            = tf.where(tf.equal(object_mask, 1), tf.maximum(ciou, tf.zeros_like(ciou)), tf.zeros_like(ciou))
        confidence_loss = K.binary_crossentropy(tobj, raw_pred[..., 4:5], from_logits=True)

        #-----------------------------------------------------------#
        #   Calculate classification loss
        #-----------------------------------------------------------#
        class_loss      = object_mask * K.binary_crossentropy(true_class_probs, raw_pred[...,5:], from_logits=True)

        #-----------------------------------------------------------#
        #   Calculate the number of positive samples
        #-----------------------------------------------------------#
        num_pos         = tf.maximum(K.sum(K.cast(object_mask, tf.float32)), 1)

        location_loss   = K.sum(ciou_loss) * box_ratio / num_pos
        confidence_loss = K.mean(confidence_loss) * balance[l] * obj_ratio
        class_loss      = K.sum(class_loss) * cls_ratio / num_pos / num_classes

        loss    += location_loss + confidence_loss + class_loss
        # if print_loss:
        # loss = tf.Print(loss, [loss, location_loss, confidence_loss, class_loss], message='loss: ')
    return loss

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
