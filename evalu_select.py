import os
import numpy as np
import argparse

#---------------------evalu.pyを拡張させる---------------------#
# コマンドライン引数の解析機能を追加
# `argparse`モジュールを使用して、引数`--evalu`を追加し、評価パラメータを指定できるようにした
#------------------------------------------------------------#
# 1. `parse_opt`関数で、コマンドライン引数を解析し、`--evalu`パラメータの値を取得
# 2. `evalu`関数を呼び出し、指定された評価パラメータに基づいて最適なモデルを選択し、その詳細情報を表示
#------------------------------------------------------------#
# コマンドライン引数を使って、評価パラメータ（例：`--evalu map`）を指定することができる
# 指定された評価パラメータに応じて、最適なモデルが選択され、その詳細情報が表示される
# 例：`python3 evalu_select.py --evalu map`
# 例：`python3 evalu_select.py --evalu flops`
# 例：`python3 evalu_select.py --evalu model_size`
# 例：`python3 evalu_select.py --evalu fps`
# 例：`python3 evalu_select.py --evalu parameter`
#------------------------------------------------------------#

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--evalu', type=str, default='', help='evaluation paramater')
    return parser.parse_known_args()[0] if known else parser.parse_args()

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
            best_index = np.argmax(fpss)
        elif evalu_param == 'parameter':
            best_index = np.argmin(parameters)

    print('The best model is: {}.'.format(model_names[best_index]))
    print('mAP:{0:.2f}%'.format(maps[best_index]),'flops:%.2fG'%(flopss[best_index]),
        'model_size:%.2fMB'%(model_sizes[best_index]), 'FPS:%.2f'%fpss[best_index],
        'Parameters:%.2fM' % parameters[best_index])
    #print('mAP:{}%'.format(maps[best_index]),'flops:%sG,'%(flopss[best_index]), 'model_size:%sMB'%(model_sizes[best_index]), 'FPS:%s'%fpss[best_index])
opt = parse_opt(True)
print(opt)
evalu(opt.evalu)
