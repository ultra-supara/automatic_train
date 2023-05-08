from distutils.util import change_root
import io
from pickle import GLOBAL
import string
import subprocess
#from sys import flags, stdout

from PIL import Image
from pathlib import Path
import cv2
from flask import Flask, flash, make_response, render_template, request, redirect
from flask_wtf import FlaskForm
from matplotlib.pyplot import flag

from wtforms import StringField, SelectMultipleField, SubmitField, IntegerField
from wtforms.validators import DataRequired, Length
from werkzeug.utils import secure_filename

import os,fnmatch
from matplotlib.font_manager import weight_dict
import yaml
import threading

# instanceの作成
app = Flask(__name__)

# 学習に関連するパスやフォルダの設定
ALLOWED_EXTENSIONS = {'txt', 'png', 'jpg', 'jpeg'}
RESULT_FOLDER = os.path.join('static')
UPLOAD_FOLDER = os.path.join('static/upload/')
CROPPED_FOLDER = os.path.join('static/cropped/')
BACKBONE_FOLDER = os.path.join('yolomodels/nets')
NECK_FOLDER = os.path.join('yolomodels/yolos')
app.config['RESULT_FOLDER'] = RESULT_FOLDER
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['BACKBONE_FOLDER'] = BACKBONE_FOLDER
app.config['NECK_FOLDER'] = NECK_FOLDER
app.config['Cropped_FOLDER'] = CROPPED_FOLDER
app.secret_key = 'secret_key'

# 畳み込みニューラルネットワークの学習に関連する選択肢
class TestForm(FlaskForm):
    backbone_names = os.listdir(app.config['BACKBONE_FOLDER'])
    backbones=[]
    for backbone_name in backbone_names:
        if  backbone_name.lower().endswith(('.py')):
            backbone_name = backbone_name.split('.')[0]
            backbones.append(backbone_name)

    net_names = os.listdir(app.config['NECK_FOLDER'])
    necks=[]
    for net_name in net_names:
        if  net_name.lower().endswith(('.py')):
            net_name = net_name.split('.')[0]
            necks.append(net_name)

    batchsize = SelectMultipleField('batchsize',
        choices=['1','2','4','6','8'], validators=[DataRequired()],)
    imagesize = SelectMultipleField('imagesize',
        choices=['32','320','416','512','640'], validators=[DataRequired()],)
    epoch = SelectMultipleField('epoch',
        choices=['1','20','50','100','150','200','300','500','1000'], validators=[DataRequired()],)
    evalu = SelectMultipleField('evalu',
        choices=['map','model_size','parameter','fps','flops'], validators=[DataRequired()],)

    backbones = SelectMultipleField(
        label='backbone', choices=backbones, validators=[DataRequired()])

    necks = SelectMultipleField(
        label='neck', choices=necks, validators=[DataRequired()])
    train = SubmitField(label='train')
    auto_train = SubmitField(label='auto_train')

    Onekey_train = SubmitField(label='Onekey_train')
    select_backbone = SubmitField(label='select_backbone')
    select_neck = SubmitField(label='select_neck')
    get_bestmodel = SubmitField(label='get_bestmodel')
    stop = SubmitField(label='stop')

# yamlファイルの操作
def change_yaml(num,name):
    with open("./data/myT.yaml")as f:
        file1 = yaml.load(f,Loader=yaml.FullLoader)

        print(file1)
        print("type : ", type(file1))
        file1['nc']=num
        file1['names'].append(name)
        with open("./data/myT"+'0913.yaml','w')as f1:
            data = yaml.dump(file1, f1)
    return file1

#
def write2txt(std):
    with open("./train.txt",'w') as f:
        for line in iter(std.readline, b''):
            print(line.decode().rstrip())
            f.write(line.decode())
            f.flush()

# subprocessの操作
def subproc(cmd):
    proc = subprocess.Popen((cmd), shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    cmd_result=''
    # for i in proc.stdout.readlines():
    #     cmd_result += i.decode()
    #     print(cmd_result)
    w1 = threading.Thread(target=write2txt,args=(proc.stdout,))
    w1.start()


class subproc1(object):
    pid = 0
    subproc = None
    w1 =None
    def __init__(self,cmd):
        self.cmd = cmd
    def run(self,cmd):
        self.cmd = cmd
        print(self.cmd)
        # self.cmd = 'exec lsof -i -r 1'
        proc = subprocess.Popen(("exec "+self.cmd), shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        subproc1.proc = proc
        subproc1.pid = proc.pid
        print(subproc1.pid)
        cmd_result=''
            # for i in proc.stdout.readlines():
            #     cmd_result += i.decode()
            #     print(cmd_result)
        subproc1.w1 = threading.Thread(target=write2txt,args=(proc.stderr,))
        print(subproc1.w1)
        subproc1.w1.start()
    def stop(self):
        print(111)
        subproc1.proc.terminate()
def allowed_file(filename):
    return '.' in filename and \
            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

#---------------------index.html---------------------#
@app.route('/', endpoint='Home', methods=['GET', 'POST'])
def home():
    form=TestForm()
    bt_a = request.values.get("train")

    bt_d = request.values.get("auto_train")

    if bt_a == "train":
        return render_template('train.html',form = form, end=True)

    if bt_d == "auto_train":
        return render_template('train.html',form = form, end=True)

    return render_template('index.html',form = form)

#---------------------train.html---------------------#
@app.route('/autotrain/', endpoint='autotrain', methods=['GET', 'POST'])
def autotrainByYaml():
    form=TestForm()

    bt_a = request.values.get("Onekey_train")
    bt_d = request.values.get("auto_train")
    bt_c = request.values.get("get_bestmodel")
    bt_d = request.values.get("stop")

    if bt_a == "Onekey_train":
        os.system('python3 train_onekey.py > static/train.txt')
        return render_template('autotrain.html',form = form, get=True)

    if bt_d == "auto_train":
        os.system('python3 train_auto.py > static/train.txt --batch %s --epochs %s --imagesize %s --evalu %s'% (form.data.get("batchsize")[0], form.data.get("epoch")[0], form.data.get("imagesize")[0], form.data.get("evalu")[0]))
        return render_template('autotrain.html',form = form, get=True)

    if bt_c == "get_bestmodel":
        os.system('python3 evalu_select.py > static/bestmodel.txt --evalu %s'% (form.data.get("evalu")[0]))
        return render_template('autotrain.html',form = form, get=True)

    if bt_d == "stop":

        return render_template('autotrain.html',form = form, get=True)

    return render_template('autotrain.html',form = form)

#---------------------manualtrain.html---------------------#
@app.route('/manualtrain/',  endpoint='manualtrain', methods=['GET', 'POST'])
def manualtrainByYaml():
    form=TestForm()

    bt_a = request.values.get("train")
    bt_c = request.values.get("get_bestmodel")
    bt_e = request.values.get("select_backbone")
    bt_f = request.values.get("select_neck")

    if bt_a == "train":
        os.system('python3 train_select.py > static/train.txt --batch %s --epochs %s --imagesize %s --neck %s  --backbone %s' % (form.data.get("batchsize")[0], form.data.get("epoch")[0], form.data.get("imagesize")[0],form.data.get("necks")[0], form.data.get("backbones")[0]))
        return render_template('manualtrain.html',form = form, end=True)

    if bt_e == "select_backbone":
        os.system('python3 train_select_backbone.py > static/train.txt --batch %s --epochs %s --imagesize %s  --backbone %s' % (form.data.get("batchsize")[0], form.data.get("epoch")[0], form.data.get("imagesize")[0], form.data.get("backbones")[0]))
        return render_template('manualtrain.html',form = form, get=True)

    if bt_f == "select_neck":
        os.system('python3 train_select_neck.py > static/train.txt --batch %s --epochs %s --imagesize %s --neck %s' % (form.data.get("batchsize")[0], form.data.get("epoch")[0], form.data.get("imagesize")[0],form.data.get("necks")[0]))
        return render_template('manualtrain.html',form = form, get=True)

    if bt_c == "get_bestmodel":
        os.system('python3 evalu_select.py > static/bestmodel.txt --evalu %s'% (form.data.get("evalu")[0]))
        return render_template('manualtrain.html',form = form, get=True)

    return render_template('manualtrain.html',form = form)

#---------------------result.html---------------------#
@app.route('/result/', endpoint='result', methods=['GET', 'POST'])
def resultByYaml():
    form=TestForm()
    bt_a = request.values.get("train")

    bt_d = request.values.get("auto_train")

    if bt_a == "train":
        return render_template('result.html',form = form, end=True)

    if bt_d == "auto_train":
        return render_template('result.html',form = form, end=True)

    return render_template('result.html',form = form)

#---------------------__main__---------------------#
if __name__ == '__main__':
        #app.run()
        app.run(host='0.0.0.0', threaded=True, port=8001)
