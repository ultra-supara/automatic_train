import math
from functools import partial

import tensorflow as tf
from tensorflow.keras import backend as K
from utils.utils_bbox import get_anchors_and_decode

# from yolo_training_v3 import convert_annotation, classes, VOCdevkit_path, VOCdevkit_sets, photo_nums, nums
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
    #   Computing Center Gap
    #   center_distance (batch, feat_w, feat_h, anchor_num)
    #-----------------------------------------------------------#
    center_distance = K.sum(K.square(b1_xy - b2_xy), axis=-1)
    enclose_mins = K.minimum(b1_mins, b2_mins)
    enclose_maxes = K.maximum(b1_maxes, b2_maxes)
    enclose_wh = K.maximum(enclose_maxes - enclose_mins, 0.0)
    #-----------------------------------------------------------#
    #   Calculate Diagonal Distance
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
#   predict_boxとreal_boxの IOU を計算
#---------------------------------------------------#
def box_iou(b1, b2):
    #---------------------------------------------------#
    #   num_anchor,1,4
    #   左上隅と右下隅の座標を計算する
    #---------------------------------------------------#
    b1          = K.expand_dims(b1, -2)
    b1_xy       = b1[..., :2]
    b1_wh       = b1[..., 2:4]
    b1_wh_half  = b1_wh/2.
    b1_mins     = b1_xy - b1_wh_half
    b1_maxes    = b1_xy + b1_wh_half

    #---------------------------------------------------#
    #   1,n,4
    #   左上隅と右下隅の座標を計算します
    #---------------------------------------------------#
    b2          = K.expand_dims(b2, 0)
    b2_xy       = b2[..., :2]
    b2_wh       = b2[..., 2:4]
    b2_wh_half  = b2_wh/2.
    b2_mins     = b2_xy - b2_wh_half
    b2_maxes    = b2_xy + b2_wh_half

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
#   lossを計算する
#---------------------------------------------------#
def yolo_loss(
    args,
    input_shape,
    anchors,
    anchors_mask,
    num_classes,
    ignore_thresh   = 0.5,
    balance         = [0.4, 1.0, 4],
    box_ratio       = 0.05,
    obj_ratio       = 1,
    cls_ratio       = 0.5 / 4,
    ciou_flag       = True,
    print_loss      = False
):
    num_layers      = len(anchors_mask)
    #---------------------------------------------------------------------------------------------------#
    #  予測結果を実際のグラウンド トゥルースから分離します。args は [*model_body.output, *y_true] です。
    #   y_true は 3つのfeature_layerを含むリストで、形状は次のとおりです。
    #   (m,13,13,3,85)
    #   (m,26,26,3,85)
    #   (m,52,52,3,85)
    #   yolo_outputs は 3つのfeature_layerを含むリストで、形状は次のとおりです。
    #   (m,13,13,3,85)
    #   (m,26,26,3,85)
    #   (m,52,52,3,85)
    #---------------------------------------------------------------------------------------------------#
    y_true          = args[num_layers:]
    yolo_outputs    = args[:num_layers]

    #-----------------------------------------------------------#
    #   Get input_shape as 416,416
    #-----------------------------------------------------------#
    input_shape = K.cast(input_shape, K.dtype(y_true[0]))
    #-----------------------------------------------------------#
    #   The shape of the grid is [13,13]; [26,26]; [52,52]
    #-----------------------------------------------------------#
    grid_shapes = [K.cast(K.shape(yolo_outputs[l])[1:3], K.dtype(y_true[0])) for l in range(num_layers)]

    #-----------------------------------------------------------#
    #   各画像を取得する
    #   mの値は batch_size です
    #-----------------------------------------------------------#
    m = K.shape(yolo_outputs[0])[0]

    loss    = 0
    #---------------------------------------------------------------------------------------------------#
    #   y_true は 3つのfeature_layerを含むリストで、形状は (m,13,13,3,85),(m,26,26,3,85),(m,52,52,3,85) です。
    #   yolo_outputs は、形状が (m,13,13,3,85),(m,26,26,3,85),(m,52,52,3,85) の 3つのfeature_layerを含むリストです。
    #---------------------------------------------------------------------------------------------------#
    for l in range(num_layers):
        #-----------------------------------------------------------#
        #   Take the first feature layer (m,13,13,3,85) as an example
        #   Get the position of the point where the target exists in the feature layer. (m,13,13,3,1)
        #-----------------------------------------------------------#
        object_mask         = y_true[l][..., 4:5]
        #-----------------------------------------------------------#
        #   Take out the corresponding type (m,13,13,3,80)
        #-----------------------------------------------------------#
        true_class_probs    = y_true[l][..., 5:]

        #-----------------------------------------------------------#
        #   yolo_outputs のfeature_layer出力を処理し、4つの戻り値を取得する
        # grid (13,13,1,2) grid座標
        # raw_pred (m,13,13,3,85) 未処理の予測結果
        # pred_xy (m,13,13,3,2) デコードされた中心座標
        # pred_w (m,13,13,3,2) デコードされた幅と高さの座標
        #-----------------------------------------------------------#
        grid, raw_pred, pred_xy, pred_wh = get_anchors_and_decode(yolo_outputs[l],
            anchors[anchors_mask[l]], num_classes, input_shape, calc_loss=True)

        #-----------------------------------------------------------#
        #   pred_box は、デコードされた予測ボックスの場所です
        #   (m,13,13,3,4)
        #-----------------------------------------------------------#
        pred_box = K.concatenate([pred_xy, pred_wh])

        #-----------------------------------------------------------#
        #   To find the negative sample group, the first step is to create an array
        #-----------------------------------------------------------#
        ignore_mask = tf.TensorArray(K.dtype(y_true[0]), size=1, dynamic_size=True)
        object_mask_bool = K.cast(object_mask, 'bool')

        #-----------------------------------------------------------#
        #   Calculate ignore_mask for each picture
        #-----------------------------------------------------------#
        def loop_body(b, ignore_mask):
            #-----------------------------------------------------------#
            #   Take out n ground truth boxes: n,4
            #-----------------------------------------------------------#
            true_box = tf.boolean_mask(y_true[l][b,...,0:4], object_mask_bool[b,...,0])
            #-----------------------------------------------------------#
            #   予測フレームと実フレームの iou を計算する
            #   Calculate the iou of the predicted frame and the real frame
            #   13,13,3,4 The coordinates of the prediction box
            #   n,4 coordinates of the ground truth box
            #   13,13,3,n The iou of the predicted frame and the real frame
            #-----------------------------------------------------------#
            iou = box_iou(pred_box[b], true_box)

            #-----------------------------------------------------------#
            #   best_iou    13,13,3 The maximum degree of coincidence between each feature point and the real box
            #-----------------------------------------------------------#
            best_iou = K.max(iou, axis=-1)

            #-----------------------------------------------------------#
            #   予測ボックスと実ボックスの最大iouがignore_threshより小さいかどうかを判定する
            #   then prediction box is considered to no real box corresponding to it
            #   特徴点の点で真のフレームに非常に近く対応する予測結果は無視する、これらのフレームはすでに比較的正確
            #   ネガティブサンプルとして扱うには適さないので無視する
            #-----------------------------------------------------------#
            ignore_mask = ignore_mask.write(b, K.cast(best_iou<ignore_thresh, K.dtype(true_box)))
            return b+1, ignore_mask

        #-----------------------------------------------------------#
        #   この場所でループを作る。ループは各イメージのために作られる
        #-----------------------------------------------------------#
        _, ignore_mask = tf.while_loop(lambda b,*args: b < m, loop_body, [0, ignore_mask])

        #-----------------------------------------------------------#
        #   ignore_maskは、特徴点をネガティブサンプルとして抽出するために使用される
        #   (m,13,13,3)
        #-----------------------------------------------------------#
        ignore_mask = ignore_mask.stack()
        #   (m,13,13,3,1)
        ignore_mask = K.expand_dims(ignore_mask, -1)

        #-----------------------------------------------------------#
        #   reshape_y_true[...,2:3]和reshape_y_true[...,3:4]
        #   これは実フレームの幅と高さを表し、いずれも0〜1 , 実フレームが大きいほど重量が少なく、小さいほど重量がある
        #-----------------------------------------------------------#
        box_loss_scale  = 2 - y_true[l][...,2:3] * y_true[l][...,3:4]
        if ciou_flag:
            #-----------------------------------------------------------#
            #   ciou_loss の計算
            #-----------------------------------------------------------#
            raw_true_box    = y_true[l][...,0:4]
            ciou            = box_ciou(pred_box, raw_true_box)
            ciou_loss       = object_mask * (1 - ciou)
            location_loss   = K.sum(ciou_loss)
        else:
            #-----------------------------------------------------------#
            #   real_boxを予測した箱と同じ形式になるようにencodeし、後でlossの計算に使用
            #-----------------------------------------------------------#
            raw_true_xy     = y_true[l][..., :2] * grid_shapes[l][::-1] - grid
            raw_true_wh     = K.log(y_true[l][..., 2:4] / anchors[anchors_mask[l]] * input_shape[::-1])

            #-----------------------------------------------------------#
            #   object_maskは、実ターゲットのwh値が存在する場合、その値を保存
            #   if/elseの条件判定文であるswitch interface
            #-----------------------------------------------------------#
            raw_true_wh     = K.switch(object_mask, raw_true_wh, K.zeros_like(raw_true_wh))
            #-----------------------------------------------------------#
            #   Use binary_crossentropy to calculate the centroid offset for better results
            #-----------------------------------------------------------#
            xy_loss         = object_mask * box_loss_scale * K.binary_crossentropy(raw_true_xy, raw_pred[...,0:2], from_logits=True)
            #-----------------------------------------------------------#
            #   wh_loss is used to calculate the width and height loss
            #-----------------------------------------------------------#
            wh_loss         = object_mask * box_loss_scale * 0.5 * K.square(raw_true_wh - raw_pred[...,2:4])
            location_loss   = (K.sum(xy_loss) + K.sum(wh_loss)) * 0.1

        #------------------------------------------------------------------------------#
        #   その場所に箱があった場合、クロスエントロピーを1、信頼度を計算する
        #   もしその場所に箱がなければ、クロスエントロピを0として信頼度を計算する
        #   この処理ではいくつかのサンプルが無視され、これらの無視されたサンプルは best_iou < ignore_thresh を満たす
        #   特徴点の点でreal_boxに非常に近く対応する予測は無視される，これらはすでに比較的正確
        #   ネガティブサンプルとして扱うには不適切なため、無視
        #------------------------------------------------------------------------------#
        confidence_loss = object_mask * K.binary_crossentropy(object_mask, raw_pred[...,4:5], from_logits=True) + \
                    (1 - object_mask) * K.binary_crossentropy(object_mask, raw_pred[...,4:5], from_logits=True) * ignore_mask

        class_loss      = object_mask * K.binary_crossentropy(true_class_probs, raw_pred[...,5:], from_logits=True)

        #-----------------------------------------------------------#
        #   Calculate the number of positive samples
        #-----------------------------------------------------------#
        num_pos         = tf.maximum(K.sum(K.cast(object_mask, tf.float32)), 1)
        num_neg         = tf.maximum(K.sum(K.cast((1 - object_mask) * ignore_mask, tf.float32)), 1)
        #-----------------------------------------------------------#
        #   Summing all losses
        #-----------------------------------------------------------#
        location_loss   = location_loss * box_ratio / num_pos
        confidence_loss = K.sum(confidence_loss) * balance[l] * obj_ratio / (num_pos + num_neg)
        class_loss      = K.sum(class_loss) * cls_ratio / num_pos / num_classes

        loss            += location_loss + confidence_loss + class_loss
        if print_loss:
            loss = tf.Print(loss, [loss, location_loss, confidence_loss, class_loss, tf.shape(ignore_mask)], summarize=100, message='loss: ')
    return loss

# ---------------------------------------------------#
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
