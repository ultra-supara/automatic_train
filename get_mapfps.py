import os
from yolo import YOLO
from yolox import YOLOx
from PIL import Image
from tqdm import tqdm
from utils.utils import get_classes
from utils.utils_map import get_map
import xml.etree.ElementTree as ET

# --------------------------------#
# 指定されたneckとbackboneを使ってYOLOのmAPとFPSを計算するための関数get_mapfps()
#   適切なデータセット構造（VOCデータセット形式）が存在し、モデルが事前に訓練されていることを前提
def get_mapfps(neck, backbone, imagesize):
    classes_path    = 'model_data/dish_classes.txt'
    MINOVERLAP      = 0.5
    confidence      = 0.001
    nms_iou         = 0.5
    score_threhold  = 0.5
    map_vis         = False
    VOCdevkit_path  = 'VOCdevkit'

    test_interval =100
    fps_path = 'img/10.jpg'

    map_out_path    = 'map_out/%s_%s'% (neck, backbone)
    image_ids = open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Main/test.txt")).read().strip().split()
    # --------------------------------#
    # get map , 必要なディレクトリが存在しない場合、それらを作成
    # --------------------------------#
    if not os.path.exists(map_out_path):
        os.makedirs(map_out_path)
    if not os.path.exists(os.path.join(map_out_path, 'ground-truth')):
        os.makedirs(os.path.join(map_out_path, 'ground-truth'))
    if not os.path.exists(os.path.join(map_out_path, 'detection-results')):
        os.makedirs(os.path.join(map_out_path, 'detection-results'))
    if not os.path.exists(os.path.join(map_out_path, 'images-optional')):
        os.makedirs(os.path.join(map_out_path, 'images-optional'))

    # --------------------------------#
    # クラス名を取得
    # --------------------------------#
    class_names, _ = get_classes(classes_path)
    # --------------------------------#
    # YOLOのモデルをロード
    # --------------------------------#
    print("Load model.")
    if neck == 'yolox':

        yolo = YOLOx(confidence = confidence, nms_iou = nms_iou, neck = neck, backbone =backbone, imagesize=imagesize)
    else:
        yolo = YOLO(confidence = confidence, nms_iou = nms_iou, neck = neck, backbone =backbone, imagesize=imagesize)
    print("Load model done.")

    print("Get predict result.")
    for image_id in tqdm(image_ids):
        image_path  = os.path.join(VOCdevkit_path, "VOC2007/JPEGImages/"+image_id+".jpg")
        image       = Image.open(image_path)
        if map_vis:
            image.save(os.path.join(map_out_path, "images-optional/" + image_id + ".jpg"))
        yolo.get_map_txt(image_id, image, class_names, map_out_path)
    print("Get predict result done.")

    print("Get ground truth result.") # 実際のアノテーションを取得する
    for image_id in tqdm(image_ids):
        with open(os.path.join(map_out_path, "ground-truth/"+image_id+".txt"), "w") as new_f:
            root = ET.parse(os.path.join(VOCdevkit_path, "VOC2007/Annotations/"+image_id+".xml")).getroot()
            for obj in root.findall('object'):
                difficult_flag = False
                if obj.find('difficult')!=None:
                    difficult = obj.find('difficult').text
                    if int(difficult)==1:
                        difficult_flag = True
                obj_name = obj.find('name').text
                if obj_name not in class_names:
                    continue
                bndbox  = obj.find('bndbox')
                left    = bndbox.find('xmin').text
                top     = bndbox.find('ymin').text
                right   = bndbox.find('xmax').text
                bottom  = bndbox.find('ymax').text

                if difficult_flag:
                    new_f.write("%s %s %s %s %s difficult\n" % (obj_name, left, top, right, bottom))
                else:
                    new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))
    print("Get ground truth result done.")

    print("Get map.")
    # --------------------------------mapの計算--------------------------------#
    mAP = get_map(MINOVERLAP, True, score_threhold = score_threhold, path = map_out_path)
    print("Get map done.")
    # --------------------------------#
    # get fps
    # --------------------------------#
    # t_flops,flops = net_flops(model, table=False)
    # --------------------------------#
    # get fps, fpsの計算
    # --------------------------------#
    img = Image.open(fps_path)
    fps_= yolo.get_FPS(img, test_interval)
    fps = 1/fps_

    return mAP, fps
