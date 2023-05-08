import math
from random import sample, shuffle

import cv2
import numpy as np
from PIL import Image
from tensorflow import keras

from utils.utilsx.utils import cvtColor, preprocess_input

class YoloDatasets(keras.utils.Sequence):
    def __init__(self, annotation_lines, input_shape, anchors, batch_size, num_classes, anchors_mask, epoch_now, epoch_length, \
                        mosaic, mixup, mosaic_prob, mixup_prob, train, special_aug_ratio = 0.7):
        self.annotation_lines   = annotation_lines
        self.length             = len(self.annotation_lines)

        self.input_shape        = input_shape
        self.anchors            = anchors
        self.batch_size         = batch_size
        self.num_classes        = num_classes
        self.anchors_mask       = anchors_mask
        self.epoch_now          = epoch_now - 1
        self.epoch_length       = epoch_length
        self.mosaic             = mosaic
        self.mosaic_prob        = mosaic_prob
        self.mixup              = mixup
        self.mixup_prob         = mixup_prob
        self.train              = train
        self.special_aug_ratio  = special_aug_ratio

    def __len__(self):
        return math.ceil(len(self.annotation_lines) / float(self.batch_size))

    def __getitem__(self, index):
        image_data  = []
        box_data    = []
        for i in range(index * self.batch_size, (index + 1) * self.batch_size):
            i           = i % self.length
            #---------------------------------------------------#
            #   訓練時にデータのランダムな拡張を行う
            #   検証時にはデータのランダムな拡張を行わない
            #---------------------------------------------------#
            if self.mosaic and self.rand() < self.mosaic_prob and self.epoch_now < self.epoch_length * self.special_aug_ratio:
                lines = sample(self.annotation_lines, 3)
                lines.append(self.annotation_lines[i])
                shuffle(lines)
                image, box = self.get_random_data_with_Mosaic(lines, self.input_shape)

                if self.mixup and self.rand() < self.mixup_prob:
                    lines           = sample(self.annotation_lines, 1)
                    image_2, box_2  = self.get_random_data(lines[0], self.input_shape, random = self.train)
                    image, box      = self.get_random_data_with_MixUp(image, box, image_2, box_2)
            else:
                image, box  = self.get_random_data(self.annotation_lines[i], self.input_shape, random = self.train)
            image_data.append(preprocess_input(np.array(image, np.float32)))
            box_data.append(box)

        image_data  = np.array(image_data)
        box_data    = np.array(box_data)
        y_true      = self.preprocess_true_boxes(box_data, self.input_shape, self.anchors, self.num_classes)
        return [image_data, *y_true], np.zeros(self.batch_size)

    def generate(self):
        i = 0
        while True:
            image_data  = []
            box_data    = []
            for b in range(self.batch_size):
                if i==0:
                    np.random.shuffle(self.annotation_lines)
                #---------------------------------------------------#
                #   訓練時にデータのランダムな拡張を行う
                #   検証時にはデータのランダムな拡張を行わない
                #---------------------------------------------------#
                if self.mosaic and self.rand() < self.mosaic_prob and self.epoch_now < self.epoch_length * self.special_aug_ratio:
                    lines = sample(self.annotation_lines, 3)
                    lines.append(self.annotation_lines[i])
                    shuffle(lines)
                    image, box = self.get_random_data_with_Mosaic(lines, self.input_shape)

                    if self.mixup and self.rand() < self.mixup_prob:
                        lines           = sample(self.annotation_lines, 1)
                        image_2, box_2  = self.get_random_data(lines[0], self.input_shape, random = self.train)
                        image, box      = self.get_random_data_with_MixUp(image, box, image_2, box_2)
                else:
                    image, box  = self.get_random_data(self.annotation_lines[i], self.input_shape, random = self.train)

                i           = (i+1) % self.length
                image_data.append(preprocess_input(np.array(image, np.float32)))
                box_data.append(box)
            image_data  = np.array(image_data)
            box_data    = np.array(box_data)
            y_true      = self.preprocess_true_boxes(box_data, self.input_shape, self.anchors, self.num_classes)
            yield image_data, y_true[0], y_true[1], y_true[2]

    def on_epoch_end(self):
        self.epoch_now += 1
        shuffle(self.annotation_lines)

    def rand(self, a=0, b=1):
        return np.random.rand()*(b-a) + a

    def get_random_data(self, annotation_line, input_shape, max_boxes=500, jitter=.3, hue=.1, sat=0.7, val=0.4, random=True):
        line    = annotation_line.split()
        #------------------------------#
        # 画像を読み込み、RGB画像に変換する
        #------------------------------#
        image   = Image.open(line[0])
        image   = cvtColor(image)
        #------------------------------#
        # 画像の高さと幅、および目標の高さと幅を取得する
        #------------------------------#
        iw, ih  = image.size
        h, w    = input_shape
        #------------------------------#
        # 予測ボックスを取得する
        #------------------------------#
        box     = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])

        if not random:
            scale = min(w/iw, h/ih)
            nw = int(iw*scale)
            nh = int(ih*scale)
            dx = (w-nw)//2
            dy = (h-nh)//2

            #---------------------------------#
            #   画像の余分な部分にグレーバーを追加する
            #---------------------------------#
            image       = image.resize((nw,nh), Image.BICUBIC)
            new_image   = Image.new('RGB', (w,h), (128,128,128))
            new_image.paste(image, (dx, dy))
            image_data  = np.array(new_image, np.float32)

            #---------------------------------#
            #  実際のボックスを調整する
            #---------------------------------#
            box_data = np.zeros((max_boxes,5))
            if len(box)>0:
                np.random.shuffle(box)
                box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
                box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
                box[:, 0:2][box[:, 0:2]<0]  = 0
                box[:, 2][box[:, 2]>w]      = w
                box[:, 3][box[:, 3]>h]      = h
                box_w   = box[:, 2] - box[:, 0]
                box_h   = box[:, 3] - box[:, 1]
                box     = box[np.logical_and(box_w>1, box_h>1)]
                if len(box)>max_boxes: box = box[:max_boxes]
                box_data[:len(box)] = box

            return image_data, box_data
        #------------------------------------------#
        #   画像を拡大縮小して幅と高さを歪ませる
        #------------------------------------------#
        new_ar = iw/ih * self.rand(1-jitter,1+jitter) / self.rand(1-jitter,1+jitter)
        scale = self.rand(.25, 2)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        image = image.resize((nw,nh), Image.BICUBIC)

        #------------------------------------------#
        #   画像の余分な部分にグレーバーを追加する
        #------------------------------------------#
        dx = int(self.rand(0, w-nw))
        dy = int(self.rand(0, h-nh))
        new_image = Image.new('RGB', (w,h), (128,128,128))
        new_image.paste(image, (dx, dy))
        image = new_image

        #------------------------------------------#
        #   画像を反転させる
        #------------------------------------------#
        flip = self.rand()<.5
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

        image_data      = np.array(image, np.uint8)
        #---------------------------------#
        #   画像の色空間を変換する
        #   色空間変換のパラメータを計算する
        #---------------------------------#
        r               = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        #---------------------------------#
        #   画像をHSVに変換する
        #---------------------------------#
        hue, sat, val   = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype           = image_data.dtype
        #---------------------------------#
        #   変換を適用する
        #---------------------------------#
        x       = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)

        #---------------------------------#
        #   実際のボックスを調整する
        #---------------------------------#
        box_data = np.zeros((max_boxes,5))
        if len(box)>0:
            np.random.shuffle(box)
            box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
            box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
            if flip: box[:, [0,2]] = w - box[:, [2,0]]
            box[:, 0:2][box[:, 0:2]<0] = 0
            box[:, 2][box[:, 2]>w] = w
            box[:, 3][box[:, 3]>h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w>1, box_h>1)] # discard invalid box
            if len(box)>max_boxes: box = box[:max_boxes]
            box_data[:len(box)] = box

        return image_data, box_data

    def merge_bboxes(self, bboxes, cutx, cuty):
        merge_bbox = []
        for i in range(len(bboxes)):
            for box in bboxes[i]:
                tmp_box = []
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

                if i == 0:
                    if y1 > cuty or x1 > cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y2 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x2 = cutx

                if i == 1:
                    if y2 < cuty or x1 > cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y1 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x2 = cutx

                if i == 2:
                    if y2 < cuty or x2 < cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y1 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x1 = cutx

                if i == 3:
                    if y1 > cuty or x2 < cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y2 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x1 = cutx
                tmp_box.append(x1)
                tmp_box.append(y1)
                tmp_box.append(x2)
                tmp_box.append(y2)
                tmp_box.append(box[-1])
                merge_bbox.append(tmp_box)
        return merge_bbox

    #---------------------------------------------------#
    '''
    データ拡張は、トレーニングデータセットを人工的に増やすことで、モデルの汎化能力を向上させるために使用されます。以下のデータ拡張手法を実装
    1. Mosaic: 4つの異なる画像をランダムに選択し、それらを1つの画像に合成します。`get_random_data_with_Mosaic`関数がこれを実装しています。
    2. MixUp: 2つの異なる画像をランダムに選択し、それらを重み付けした和でブレンドします。`get_random_data_with_MixUp`関数がこれを実装しています。
    3. ランダムフリップ: 画像を水平方向にランダムに反転させます。バウンディングボックスも反転に応じて更新
    4. ランダムスケールとアスペクト比ジッター: 画像のサイズとアスペクト比をランダムに変更します。バウンディングボックスも変更に応じて更新
    5. 色空間の変換: 画像の色相、彩度、明度をランダムに変更する

    物体検出モデルの学習に役立つように、これらのデータ拡張手法を適用した後、アノテーションをしてモデルが予測できる形式に変換します。これは、`preprocess_true_boxes`関数で実装されています。
    これらのデータ拡張手法は、トレーニングデータセットに多様性をもたらすことで、モデルがさまざまな状況で物体を検出する能力を向上させることができます。
    '''
    #---------------------------------------------------#
    #  get_random_data_with_Mosaicは、画像とバウンディングボックスデータを入力として受け取り、
    #  Mosaicデータ拡張（4つの画像を1つに結合）を行い、拡張された画像と更新されたバウンディングボックスデータを返します
    #---------------------------------------------------#
    def get_random_data_with_Mosaic(self, annotation_line, input_shape, max_boxes=500, jitter=0.3, hue=.1, sat=0.7, val=0.4):
        h, w = input_shape
        min_offset_x = self.rand(0.3, 0.7)
        min_offset_y = self.rand(0.3, 0.7)

        image_datas = []
        box_datas   = []
        index       = 0
        for line in annotation_line:
            #---------------------------------#
            #   各行を分割する
            #---------------------------------#
            line_content = line.split()
            #---------------------------------#
            #   画像を開く
            #---------------------------------#
            image = Image.open(line_content[0])
            image = cvtColor(image)
            #---------------------------------#
            #   画像のサイズ
            #---------------------------------#
            iw, ih = image.size
            #---------------------------------#
            #   ボックスの位置を保存
            #---------------------------------#
            box = np.array([np.array(list(map(int, box.split(',')))) for box in line_content[1:]])
            #---------------------------------#
            #   画像を反転させるかどうか
            #---------------------------------#
            flip = self.rand()<.5
            if flip and len(box)>0:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                box[:, [0,2]] = iw - box[:, [2,0]]

            #------------------------------------------#
            #   画像をスケーリングし、幅と高さを歪ませる
            #------------------------------------------#
            new_ar = iw/ih * self.rand(1-jitter,1+jitter) / self.rand(1-jitter,1+jitter)
            scale = self.rand(.4, 1)
            if new_ar < 1:
                nh = int(scale*h)
                nw = int(nh*new_ar)
            else:
                nw = int(scale*w)
                nh = int(nw/new_ar)
            image = image.resize((nw, nh), Image.BICUBIC)

            #-----------------------------------------------#
            #   画像を配置し、4つの分割画像の位置にそれぞれ対応させる
            #-----------------------------------------------#
            if index == 0:
                dx = int(w*min_offset_x) - nw
                dy = int(h*min_offset_y) - nh
            elif index == 1:
                dx = int(w*min_offset_x) - nw
                dy = int(h*min_offset_y)
            elif index == 2:
                dx = int(w*min_offset_x)
                dy = int(h*min_offset_y)
            elif index == 3:
                dx = int(w*min_offset_x)
                dy = int(h*min_offset_y) - nh

            new_image = Image.new('RGB', (w,h), (128,128,128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image)

            index = index + 1
            box_data = []
            #---------------------------------#
            #   Reprocess the box according to the new image
            #---------------------------------#
            if len(box)>0:
                np.random.shuffle(box)
                box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
                box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
                box[:, 0:2][box[:, 0:2]<0] = 0
                box[:, 2][box[:, 2]>w] = w
                box[:, 3][box[:, 3]>h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w>1, box_h>1)]
                box_data = np.zeros((len(box),5))
                box_data[:len(box)] = box

            image_datas.append(image_data)
            box_datas.append(box_data)

        #---------------------------------#
        #   像を分割し、一緒に配置する
        #---------------------------------#
        cutx = int(w * min_offset_x)
        cuty = int(h * min_offset_y)

        new_image = np.zeros([h, w, 3])
        new_image[:cuty, :cutx, :] = image_datas[0][:cuty, :cutx, :]
        new_image[cuty:, :cutx, :] = image_datas[1][cuty:, :cutx, :]
        new_image[cuty:, cutx:, :] = image_datas[2][cuty:, cutx:, :]
        new_image[:cuty, cutx:, :] = image_datas[3][:cuty, cutx:, :]

        new_image       = np.array(new_image, np.uint8)
        #---------------------------------#
        #   画像の色空間を変換する
        #   色空間変換のパラメータを計算する
        #---------------------------------#
        r               = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        #---------------------------------#
        #   画像をHSVに変換する
        #---------------------------------#
        hue, sat, val   = cv2.split(cv2.cvtColor(new_image, cv2.COLOR_RGB2HSV))
        dtype           = new_image.dtype
        #---------------------------------#
        #   変換を適用する
        #---------------------------------#
        x       = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        new_image = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        new_image = cv2.cvtColor(new_image, cv2.COLOR_HSV2RGB)

        #---------------------------------#
        #   実際のボックスを調整する
        #---------------------------------#
        new_boxes = self.merge_bboxes(box_datas, cutx, cuty)

        #---------------------------------#
        #   ボックスを調整する
        #---------------------------------#
        box_data = np.zeros((max_boxes, 5))
        if len(new_boxes)>0:
            if len(new_boxes)>max_boxes: new_boxes = new_boxes[:max_boxes]
            box_data[:len(new_boxes)] = new_boxes
        return new_image, box_data

    #---------------------------------------------------#
    #   get_random_data_with_MixUpは、2つの画像とそれらのバウンディングボックスを入力として受け取り、
    #   MixUpデータ拡張（2つの画像を一定の比率でブレンド）を行い、ブレンドされた画像と更新されたバウンディングボックスデータを返します。
    #---------------------------------------------------#
    def get_random_data_with_MixUp(self, image_1, box_1, image_2, box_2, max_boxes=500):
        new_image = np.array(image_1, np.float32) * 0.5 + np.array(image_2, np.float32) * 0.5

        box_1_wh    = box_1[:, 2:4] - box_1[:, 0:2]
        box_1_valid = box_1_wh[:, 0] > 0

        box_2_wh    = box_2[:, 2:4] - box_2[:, 0:2]
        box_2_valid = box_2_wh[:, 0] > 0

        new_boxes = np.concatenate([box_1[box_1_valid, :], box_2[box_2_valid, :]], axis=0)
        #---------------------------------#
        #   ボックスを調整する
        #---------------------------------#
        box_data = np.zeros((max_boxes, 5))
        if len(new_boxes)>0:
            if len(new_boxes)>max_boxes: new_boxes = new_boxes[:max_boxes]
            box_data[:len(new_boxes)] = new_boxes
        return new_image, box_data

    #---------------------------------------------------#
    #   preprocess_true_boxesは、引数を入力として受け取り、YOLOのトレーニングに使用されるtrue_boxesを前処理します。
    #   モデルに適切な形式で処理されたバウンディングボックスデータを返します。
    #---------------------------------------------------#
    def preprocess_true_boxes(self, true_boxes, input_shape, anchors, num_classes):
        assert (true_boxes[..., 4]<num_classes).all(), 'class id must be less than num_classes'
        #-----------------------------------------------------------#
        #   ボックスの座標と画像のサイズを取得する
        #-----------------------------------------------------------#
        true_boxes  = np.array(true_boxes, dtype='float32')
        input_shape = np.array(input_shape, dtype='int32')
        #-----------------------------------------------------------#
        #   合計で3つの特徴層があります
        #-----------------------------------------------------------#
        num_layers  = len(self.anchors_mask)
        #-----------------------------------------------------------#
        #   mは画像の数であり、grid_shapesはグリッドの形状です
        #-----------------------------------------------------------#
        m           = true_boxes.shape[0]
        grid_shapes = [input_shape // {0:32, 1:16, 2:8}[l] for l in range(num_layers)]
        #-----------------------------------------------------------#
        #   y_trueのフォーマットは(m,13,13,3,85)(m,26,26,3,85)(m,52,52,3,85)です
        #-----------------------------------------------------------#
        y_true = [np.zeros((m, grid_shapes[l][0], grid_shapes[l][1], len(self.anchors_mask[l]), 5 + num_classes),
                    dtype='float32') for l in range(num_layers)]

        #-----------------------------------------------------------#
        # 実際のボックスの中心と幅・高さを計算で求めます
        # 中心点(m, n, 2) 幅・高さ(m, n, 2)
        #-----------------------------------------------------------#
        boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
        boxes_wh =  true_boxes[..., 2:4] - true_boxes[..., 0:2]
        #-----------------------------------------------------------#
        # 実際のボックスを小数形式に正規化する
        #-----------------------------------------------------------#
        true_boxes[..., 0:2] = boxes_xy / input_shape[::-1]
        true_boxes[..., 2:4] = boxes_wh / input_shape[::-1]

        #-----------------------------------------------------------#
        #   [9,2] -> [1,9,2]
        #-----------------------------------------------------------#
        anchors         = np.expand_dims(anchors, 0)
        anchor_maxes    = anchors / 2.
        anchor_mins     = -anchor_maxes

        #-----------------------------------------------------------#
        #   幅と高さが0より大きい場合のみ有効です。
        #-----------------------------------------------------------#
        valid_mask = boxes_wh[..., 0]>0

        for b in range(m):
            #-----------------------------------------------------------#
            #   各画像に対して処理を行う
            #-----------------------------------------------------------#
            wh = boxes_wh[b, valid_mask[b]]
            if len(wh) == 0: continue
            #-----------------------------------------------------------#
            #   [n,2] -> [n,1,2]
            #-----------------------------------------------------------#
            wh          = np.expand_dims(wh, -2)
            box_maxes   = wh / 2.
            box_mins    = - box_maxes

            #-----------------------------------------------------------#
            #   すべての実際のボックスと事前に設定されたボックスのIoU（Intersection over Union）を計算する
            #   intersect_area  [n,9]
            #   box_area        [n,1]
            #   anchor_area     [1,9]
            #   iou             [n,9]
            #-----------------------------------------------------------#
            intersect_mins  = np.maximum(box_mins, anchor_mins)
            intersect_maxes = np.minimum(box_maxes, anchor_maxes)
            intersect_wh    = np.maximum(intersect_maxes - intersect_mins, 0.)
            intersect_area  = intersect_wh[..., 0] * intersect_wh[..., 1]

            box_area    = wh[..., 0] * wh[..., 1]
            anchor_area = anchors[..., 0] * anchors[..., 1]

            iou = intersect_area / (box_area + anchor_area - intersect_area)

            best_anchor = np.argmax(iou, axis=-1)

            for t, n in enumerate(best_anchor):
                #-----------------------------------------------------------#
                # floorは下向きの丸めで、実際のボックスが属する特徴レイヤーの対応するx、y軸座標を見つけるために使用されます。
                #-----------------------------------------------------------#
                i = np.floor(true_boxes[b, t, 0] * grid_shapes[l][1]).astype('int32')
                j = np.floor(true_boxes[b, t, 1] * grid_shapes[l][0]).astype('int32')
                #-----------------------------------------------------------#
                # kは、現在のこの特徴点のk番目の事前に設定されたボックスを指します。
                #-----------------------------------------------------------#
                k = self.anchors_mask[l].index(n)
                #-----------------------------------------------------------#
                # cは、現在のこの実際のボックスのカテゴリを指します。
                #-----------------------------------------------------------#
                c = true_boxes[b, t, 4].astype('int32')
                #-----------------------------------------------------------#
                # y_trueの形状は、(m,13,13,3,85)(m,26,26,3,85)(m,52,52,3,85)です。
                # 最後の85は、4+1+80に分解できます。4はボックスの中心と幅・高さを表し、
                # 1は信頼度を表し、80はカテゴリを表しています。
                #-----------------------------------------------------------#
                y_true[l][b, j, i, k, 0:4] = true_boxes[b, t, 0:4]
                y_true[l][b, j, i, k, 4] = 1
                y_true[l][b, j, i, k, 5+c] = 1

        return y_true
