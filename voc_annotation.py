import os
import random
import xml.etree.ElementTree as ET

import numpy as np


from utils.utils import get_classes

#--------------------------------------------------------------------------------------------------------------------------------#
#   annotation_modeは、このファイルが実行時に計算する内容を指定するために使用されます。
#   annotation_modeが0の場合、完全なラベル処理プロセスが含まれます。
#   これには、VOCdevkit/VOC2007/ImageSets内のtxtや、トレーニング用の2007_train.txt、2007_val.txtが含まれます。
#   annotation_modeが1の場合、VOCdevkit/VOC2007/ImageSets内のtxtを取得します。
#   annotation_modeが2の場合、トレーニング用の2007_train.txt、2007_val.txtを取得します。
#--------------------------------------------------------------------------------------------------------------------------------#
annotation_mode     = 0
#-------------------------------------------------------------------#
#   修正が必要です。2007_train.txtと2007_val.txtに目的の情報を生成するために使用されます。
#   トレーニングと予測に使用されるclasses_pathと一致させるだけでOKです。
#   生成された2007_train.txtに目的の情報がない場合、それはclassesが正しく設定されていないためです。
#   これは、annotation_modeが0と2の場合にのみ有効です。
#-------------------------------------------------------------------#
classes_path        = 'model_data/dish_classes.txt'
#--------------------------------------------------------------------------------------------------------------------------------#
#   trainval_percentは、(訓練データ+検証データ)とテストデータの比率を指定するために使用されます。
#   デフォルトでは、(訓練データ+検証データ)：テストデータ = 9：1です。
#   また、train_percentは、(訓練データ+検証データ)の中で訓練データと検証データの比率を指定するために使用されます。
#   デフォルトでは、訓練データ：検証データ = 9：1です。これらの設定は、annotation_modeが0および1の場合にのみ有効です。
#--------------------------------------------------------------------------------------------------------------------------------#
trainval_percent    = 0.9
train_percent       = 0.9
#-------------------------------------------------------#
#   VOCデータセットがあるフォルダを指定します。
#   デフォルトでは、ルートディレクトリの下のVOCデータセットを指しています。
#-------------------------------------------------------#
VOCdevkit_path  = 'VOCdevkit'

VOCdevkit_sets  = [('2007', 'train'), ('2007', 'val')]
classes, _      = get_classes(classes_path)

#-------------------------------------------------------#
#   ターゲット数を集計します。
#-------------------------------------------------------#
photo_nums  = np.zeros(len(VOCdevkit_sets))
nums        = np.zeros(len(classes))
#-------------------------------------------------------#
#   VOCdevkit/VOC2007/ImageSets/Mainにtxtを生成します。
def convert_annotation(year, image_id, list_file):
    in_file = open(os.path.join(VOCdevkit_path, 'VOC%s/Annotations/%s.xml'%(year, image_id)), encoding='utf-8')
    tree=ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = 0
        if obj.find('difficult')!=None:
            difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(float(xmlbox.find('xmin').text)), int(float(xmlbox.find('ymin').text)), int(float(xmlbox.find('xmax').text)), int(float(xmlbox.find('ymax').text)))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))

        nums[classes.index(cls)] = nums[classes.index(cls)] + 1

if __name__ == "__main__":
    random.seed(0)
    if " " in os.path.abspath(VOCdevkit_path):
        raise ValueError("データセットが保存されているフォルダのパスと画像名に空白が含まれていると、モデルの正常な学習に影響を与えますので修正してください")

    if annotation_mode == 0 or annotation_mode == 1:
        print("Generate txt in ImageSets.")
        xmlfilepath     = os.path.join(VOCdevkit_path, 'VOC2007/Annotations')
        saveBasePath    = os.path.join(VOCdevkit_path, 'VOC2007/ImageSets/Main')
        temp_xml        = os.listdir(xmlfilepath)
        total_xml       = []
        for xml in temp_xml:
            if xml.endswith(".xml"):
                total_xml.append(xml)

        num     = len(total_xml)
        list    = range(num)
        tv      = int(num*trainval_percent)
        tr      = int(tv*train_percent)
        trainval= random.sample(list,tv)
        train   = random.sample(trainval,tr)
        test = num-tv
        v = tv-tr
        print("all images:",num)
        #print("train and val images:",tv)
        print("train images:",tr)
        print("val images:",v)
        print("test images:",test)
        ftrainval   = open(os.path.join(saveBasePath,'trainval.txt'), 'w')
        ftest       = open(os.path.join(saveBasePath,'test.txt'), 'w')
        ftrain      = open(os.path.join(saveBasePath,'train.txt'), 'w')
        fval        = open(os.path.join(saveBasePath,'val.txt'), 'w')

        for i in list:
            name=total_xml[i][:-4]+'\n'
            if i in trainval:
                ftrainval.write(name)
                if i in train:
                    ftrain.write(name)
                else:
                    fval.write(name)
            else:
                ftest.write(name)

        ftrainval.close()
        ftrain.close()
        fval.close()
        ftest.close()
        print("Generate txt in ImageSets done.")

    if annotation_mode == 0 or annotation_mode == 2:
        print("Generate 2007_train.txt and 2007_val.txt for train.")
        type_index = 0
        for year, image_set in VOCdevkit_sets:
            image_ids = open(os.path.join(VOCdevkit_path, 'VOC%s/ImageSets/Main/%s.txt'%(year, image_set)), encoding='utf-8').read().strip().split()
            list_file = open('%s_%s.txt'%(year, image_set), 'w', encoding='utf-8')
            for image_id in image_ids:
                list_file.write('%s/VOC%s/JPEGImages/%s.jpg'%(os.path.abspath(VOCdevkit_path), year, image_id))

                convert_annotation(year, image_id, list_file)
                list_file.write('\n')
            photo_nums[type_index] = len(image_ids)
            type_index += 1
            list_file.close()
        print("Generate 2007_train.txt and 2007_val.txt for train done.")

        def printTable(List1, List2):
            for i in range(len(List1[0])):
                print("|", end=' ')
                for j in range(len(List1)):
                    print(List1[j][i].rjust(int(List2[j])), end=' ')
                    print("|", end=' ')
                print()

        str_nums = [str(int(x)) for x in nums]
        tableData = [
            classes, str_nums
        ]
        colWidths = [0]*len(tableData)
        len1 = 0
        for i in range(len(tableData)):
            for j in range(len(tableData[i])):
                if len(tableData[i][j]) > colWidths[i]:
                    colWidths[i] = len(tableData[i][j])
        printTable(tableData, colWidths)

        if photo_nums[0] <= 500:
            print("dataset too small, bigger (Epoch) to fit (Step)")

        if np.sum(nums) == 0:
            print("No object! Please revise classes_path!")
            print("No object! Please revise classes_path!")
            print("No object! Please revise classes_path!")
