import os
import numpy as np

#---------------------evalu---------------------#
""" 様々な評価指標に基づいて最適なモデルを選択する機能を提供します。
`evalu`関数は、指定された評価パラメータ（`evalu_param`）に応じて、最適なモデルを選択し、その情報を表示します。

1. テキストファイル（`static/evalu_parames.txt`）から各モデルの評価パラメータを読み込みます。
2. 各行からモデル名、mAP、FLOPS、モデルサイズ、FPS、パラメータ数を抽出し、対応するリストに追加します。
3. `evalu_param`に応じて、最適なモデルのインデックスを計算します（最大mAP、最小FLOPS、最小モデルサイズ、最大FPS、最小パラメータ数）
4. 最適なモデルの名前と評価パラメータを表示します。

異なる評価指標に基づいて最適なモデルを選択し、その詳細情報を表示することができます。これにより、特定の要件に適したモデルを簡単に特定できます。 """
#-------------------------------------------------#

def evalu(evalu_param):
    with open('static/evalu_parames.txt','r+',encoding='utf-8') as filetxt:
        lines=filetxt.readlines()
        #print(lines)

        model_names = []
        maps = []
        flopss = []
        model_sizes = []
        fpss = []
        parameters = []

        for line in lines:
            model_name = line.split(',')[0]
            model_names.append(model_name)

            map = line.split(',')[1].split(':')[1].split('%')[0]
            map = np.array(map,dtype=np.float)
            maps.append(map)

            flops= line.split(',')[2].split(':')[1].split('G')[0]
            flops = np.array(flops,dtype=np.float)
            flopss.append(flops)

            model_size = line.split(',')[3].split(':')[1].split('M')[0]
            model_size = np.array(model_size,dtype=np.float)
            model_sizes.append(model_size)

            fps = line.split(',')[4].split(':')[1]
            fps = np.array(fps,dtype=np.float)
            fpss.append(fps)

            parameter = line.split(',')[5].split(':')[1].split('M')[0]
            parameter = np.array(parameter,dtype=np.float)
            parameters.append(parameter)

        if evalu_param=='map':
            best_index = np.argmax(maps)
        elif evalu_param=='flops':
            best_index = np.argmin(flopss)
        elif evalu_param == 'model_size':
            best_index = np.argmin(model_sizes)
        elif evalu_param == 'fps':
            best_index = np.argmin(fpss)
        elif evalu_param == 'parameter':
            best_index = np.argmin(parameters)

    print('The best model is: {}.'.format(model_names[best_index]))
    print('mAP:{0:.2f}%'.format(maps[best_index]),'flops:%.2fG'%(flopss[best_index]),
        'model_size:%.2fMB'%(model_sizes[best_index]), 'FPS:%.2f'%fpss[best_index],
        'Parameters:%.2fM' % parameters[best_index])
    #print('mAP:{}%'.format(maps[best_index]),'flops:%sG,'%(flopss[best_index]), 'model_size:%sMB'%(model_sizes[best_index]), 'FPS:%s'%fpss[best_index])
