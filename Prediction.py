# 指定されたディレクトリにある画像ファイルを読み込み、物体を検出して、結果を別のディレクトリに保存する
import tensorflow as tf
from PIL import Image,ImageOps

from yolo import YOLO
import os

from tqdm import tqdm
import shutil
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

if __name__ == "__main__":

    yolo = YOLO()

    crop            = False
    count           = False
    dir_origin_path = "static/upload/"
    dir_copy_path   = "static/img_copy/"
    dir_save_path   = "static/img_out/"

    while True:
        img_names = os.listdir(dir_origin_path)

        for img_name in img_names:
            if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path  = os.path.join(dir_origin_path, img_name)
                try:
                    image       = Image.open(image_path)
                    image = ImageOps.exif_transpose(image)
                    scale=0.6
                    w,h = image.size
                    nw = int(w*scale)
                    nh = int(h*scale)
                    image = image.resize((nw, nh))
                    print(image)
                    r_image     = yolo.detect_image(image,  img_name)

    #                if not os.path.exists(dir_save_path):
    #                     os.makedirs(dir_save_path)

                    r_image.save(os.path.join(dir_save_path, img_name.replace(".png", ".jpg")), quality=95, subsampling=0)

                    shutil.copy(image_path, dir_copy_path)

                    os.remove(image_path)
                except:
                    #print('image error')
                    continue

""" 各画像ファイルに対して、次の処理を実行
- 画像を開いてリサイズし、必要に応じてExif情報に基づいて回転
- YOLOを使って画像内の物体を検出し、検出結果が描画された画像を取得
- 検出結果が描画された画像を dir_save_path に保存
- 元の画像を dir_copy_path にコピーし、dir_origin_path から削除
"""
