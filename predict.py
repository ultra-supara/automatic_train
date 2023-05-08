#-----------------------------------------------------------------------#
#   predict.py は、単一画像予測、カメラ検出、FPS テスト、ディレクトリ トラバーサル検出などの機能を統合します
#   ファイルに機能が統合されており、モードを指定することでモードが変更されます。
#   **概要**
# 複数の物体検出タスクを実行するためのスクリプトです。指定されたmodeに応じて、以下の機能を実行する。

# 1. **predict** : ユーザーが入力した画像ファイル名を使って、画像の物体検出を行います。検出された物体は、バウンディングボックスとクラスラベルが描画された画像として表示されます。

# 2. **video** : カメラや指定されたビデオファイルからリアルタイムでフレームを取得し、物体検出を行います。
# 検出された物体は、バウンディングボックスとクラスラベルが描画されたビデオとして表示されます。また、処理後のビデオを保存することもできます。

# 3. **FPS** : 指定された画像に対して、物体検出のパフォーマンスを計測します。モデルのFPSを計算し、結果を表示します。

# 4. **ディレクトリ予測**：指定されたディレクトリ内のすべての画像に対して、物体検出を行います。検出された物体は、バウンディングボックスとクラスラベルが描画された画像として保存されます。

# 5. **heatmap**：ユーザーが入力した画像ファイル名を使って、物体検出のヒートマップを作成します。ヒートマップは、指定されたパスに保存されます。
#-----------------------------------------------------------------------#
import time

import cv2
import os
import numpy as np
import tensorflow as tf
from PIL import Image

from yolo import YOLO

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

if __name__ == "__main__":
    yolo = YOLO()
    # predict, video, fps, dir_predict, heatmap#
    mode = "dir_predict"
    crop            = False
    count           = False

    video_path      = 0
    video_save_path = ""
    video_fps       = 25.0

    test_interval   = 100
    fps_image_path  = "img/010.jpg"

    dir_origin_path = "image_pre/img/"
    dir_save_path   = "image_pre/img_out/"

    heatmap_save_path = "model_data/heatmap_vision.png"

    if mode == "predict":

        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = yolo.detect_image(image, crop = crop, count = count)
                r_image.show()

    elif mode == "video":
        capture = cv2.VideoCapture(video_path)
        if video_save_path!="":
            fourcc  = cv2.VideoWriter_fourcc(*'XVID')
            size    = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out     = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

        ref, frame = capture.read()
        if not ref:
            raise ValueError("カメラを正しく読み取れませんでした。カメラが正しくインストールされているかパスを確認してください。")

        fps = 0.0
        while(True):
            t1 = time.time()
            # 特定のフレームを読み取る
            ref, frame = capture.read()
            if not ref:
                break
            # 形式を変換：BGRからRGBへ
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            # Imageに変換
            frame = Image.fromarray(np.uint8(frame))
            # 検出を行う
            frame = np.array(yolo.detect_image(frame))
            # 表示形式に合わせてRGBからBGRへ変換（OpenCV用）
            frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
            # フレームレートを計算する
            fps  = ( fps + (1./(time.time()-t1)) ) / 2
            print("fps= %.2f"%(fps))
            frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("video",frame)
            c= cv2.waitKey(1) & 0xff
            if video_save_path!="":
                out.write(frame)

            if c==27:
                capture.release()
                break

        print("Video Detection Done!")
        capture.release()
        if video_save_path!="":
            print("Save processed video to the path :" + video_save_path)
            out.release()
        cv2.destroyAllWindows()

    elif mode == "fps":
        img = Image.open(fps_image_path)
        tact_time = yolo.get_FPS(img, test_interval)
        print(str(tact_time) + ' seconds, ' + str(1/tact_time) + 'FPS, @batch_size 1')

    elif mode == "dir_predict":
        import os
        num = 0
        img_names = os.listdir(dir_origin_path)
        for img_name in img_names:
            if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path  = os.path.join(dir_origin_path, img_name)
                image       = Image.open(image_path)
                r_image     = yolo.detect_image(image,img_name)
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                r_image.save(os.path.join(dir_save_path, img_name.replace(".jpg", ".png")), quality=95, subsampling=0)
                num += 1
        print('{} images have been predicted.'.format(num))

    elif mode == "heatmap":
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                yolo.detect_heatmap(image, heatmap_save_path)

    else:
        raise AssertionError("Please specify the correct mode: 'predict', 'video', 'fps' or 'dir_predict'.")
