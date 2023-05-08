
import os
import datetime
import tensorflow as tf
from tensorflow.keras.callbacks import (EarlyStopping, LearningRateScheduler,
                                        TensorBoard)
from tensorflow.keras.optimizers import SGD, Adam

from yolomodels.yolos.yolov3 import get_train_modelv3, yolov3_body
from yolomodels.yolos.yolov4 import get_train_modelv4, yolov4_body
from yolomodels.yolos.yolov5 import get_train_modelv5, yolov5_body
from yolomodels.yolos.yolov7 import get_train_modelv7, yolov7_body
from yolomodels.yolos.yolox import get_train_modelx, yolox_body

from get_mapfps import get_mapfps
from utils.utils import get_classes
from get_params import get_params
import argparse
import shutil


def log_model_summary(text):
    with open('modelsummary.txt', 'a') as f:
        f.write(text)
        f.write('\n')
    f.close()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 学習パラメタの設定
if __name__ == "__main__":
    imagesize = 32  #N*32(N=(1,20))
    epochs = 50
    batch_size = 2
    evalu_param = 'map'
    eager           = False
    train_gpu       = [0,]
    classes_path    = 'model_data/voc_classes.txt'
    anchors_path    = 'model_data/yolo_anchors.txt'
    anchors_mask    = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    model_path      = ''

    input_shape     = [imagesize, imagesize]
    mosaic              = True
    mosaic_prob         = 0.5
    mixup               = True
    mixup_prob          = 0.5
    special_aug_ratio   = 0.7

    #label_smoothing     = 0

    Init_Epoch          = 0
    UnFreeze_Epoch      = epochs
    Unfreeze_batch_size = batch_size

    Freeze_Train        = False

    Init_lr             = 1e-3
    Min_lr              = Init_lr * 0.01

    optimizer_type      = "adam"
    momentum            = 0.937
    weight_decay        = 0

    lr_decay_type       = 'cos'

    focal_loss          = False
    focal_alpha         = 0.25
    focal_gamma         = 2

    save_period         = 10

    eval_flag           = True
    eval_period         = 20

    num_workers         = 1

    train_annotation_path   = '2007_train.txt'
    val_annotation_path     = '2007_val.txt'

    os.environ["CUDA_VISIBLE_DEVICES"]  = ','.join(str(x) for x in train_gpu)
    ngpus_per_node                      = len(train_gpu)

    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    if ngpus_per_node > 1 and ngpus_per_node > len(gpus):
        raise ValueError("The number of GPUs specified for training is more than the GPUs on the machine")

    if ngpus_per_node > 1:
        strategy = tf.distribute.MirroredStrategy()
    else:
        strategy = None
    print('Number of devices: {}'.format(ngpus_per_node))

    num = 0
    num_neck = 0
    num_backbone = 0
    if os.path.exists('static/evalu_parames.txt'):
        os.remove('static/evalu_parames.txt')
    dir_yolos_path = "yolomodels/yolos"
    yolos_names = os.listdir(dir_yolos_path)
    for yolos_name in yolos_names:
        if yolos_name.lower().endswith(('.py')):
            yolos_name = yolos_name.split('.')[0]
            print('There are %s necks.'% len(yolos_names))
            #print(yolos_name)
            neck = yolos_name

            if neck == 'yolov3':
                yolo_body = yolov3_body
                get_train_model = get_train_modelv3
                from utils.utilsv3.callbacks import EvalCallback, LossHistory, ModelCheckpoint
                from utils.utilsv3.dataloader import YoloDatasets
                from utils.utilsv3.utils import get_anchors, get_classes, show_config, net_flops
                from yolomodels.yolo_training_v3 import get_lr_scheduler
            elif neck == 'yolov4':
                yolo_body=yolov4_body
                get_train_model = get_train_modelv4
                from utils.utilsv4.callbacks import EvalCallback, LossHistory, ModelCheckpoint
                from utils.utilsv4.dataloader import YoloDatasets
                from utils.utilsv4.utils import get_anchors, get_classes, show_config, net_flops
                from yolomodels.yolo_training_v4 import get_lr_scheduler
            elif neck == 'yolov5':
                yolo_body=yolov5_body
                get_train_model = get_train_modelv5
                from utils.utilsv5.callbacks import EvalCallback, LossHistory, ModelCheckpoint
                from utils.utilsv5.dataloader import YoloDatasets
                from utils.utilsv5.utils import get_anchors, get_classes, show_config, net_flops
                from yolomodels.yolo_training_v5 import get_lr_scheduler
            elif neck == 'yolov7':
                yolo_body=yolov7_body
                get_train_model = get_train_modelv7
                from utils.utilsv7.callbacks import EvalCallback, LossHistory, ModelCheckpoint
                from utils.utilsv7.dataloader import YoloDatasets
                from utils.utilsv7.utils import get_anchors, get_classes, show_config, net_flops
                from yolomodels.yolo_training_v7 import get_lr_scheduler
            elif neck == 'yolox':
                yolo_body=yolox_body
                get_train_model = get_train_modelx
                from utils.utilsx.callbacks import EvalCallback, LossHistory, ModelCheckpoint
                from utils.utilsx.dataloader import YoloDatasets
                from utils.utilsx.utils import get_classes, show_config, net_flops
                from yolomodels.yolo_training_x import get_lr_scheduler

            if neck == 'yolox':
                class_names, num_classes = get_classes(classes_path)
            else:
                class_names, num_classes = get_classes(classes_path)
                anchors, num_anchors     = get_anchors(anchors_path)

            num_backbone = 0

            dir_nets_path = "yolomodels/nets"
            nets_names = os.listdir(dir_nets_path)
            for nets_name in nets_names:
                if nets_name.lower().endswith(('.py')):
                    nets_name = nets_name.split('.')[0]
                    print('There are %s backbones.'% len(nets_names))
                    #print(nets_name)
                    backbone = nets_name
                    print('Current training model: {} +'.format(neck),'{}!'.format(backbone))
                    print('Evaluation parameter: mAP')
                    if os.path.exists('modelsummary.txt'):
                        os.remove('modelsummary.txt')
                    if ngpus_per_node > 1:
                        with strategy.scope():
                            if neck == 'yolox':
                                x = input_shape[0]
                                model_body  = yolo_body((x, x, 3), num_classes, backbone)
                                if model_path != '':
                                    print('Load weights {}.'.format(model_path))
                                    model_body.load_weights(model_path, by_name=True, skip_mismatch=True)
                                if not eager:
                                    model = get_train_model(model_body, input_shape, num_classes)
                            else:
                                x = input_shape[0]
                                model_body  = yolo_body((x, x, 3), anchors_mask, num_classes, backbone)
                                if model_path != '':
                                    print('Load weights {}.'.format(model_path))
                                    model_body.load_weights(model_path, by_name=True, skip_mismatch=True)
                                if not eager:
                                    model = get_train_model(model_body, input_shape, num_classes, anchors, anchors_mask)
                    else:
                        if neck == 'yolox':
                            x = input_shape[0]
                            model_body  = yolo_body((x, x, 3), num_classes, backbone)
                            if model_path != '':
                                print('Load weights {}.'.format(model_path))
                                model_body.load_weights(model_path, by_name=True, skip_mismatch=True)
                            if not eager:
                                model = get_train_model(model_body, input_shape, num_classes)
                        else:
                            x = input_shape[0]
                            model_body  = yolo_body((x, x, 3), anchors_mask, num_classes, backbone)
                            if model_path != '':
                                print('Load weights {}.'.format(model_path))
                                model_body.load_weights(model_path, by_name=True, skip_mismatch=True)
                            if not eager:
                                model = get_train_model(model_body, input_shape, num_classes, anchors, anchors_mask)
                    t_flops,flops = net_flops(model_body, table=False)
                    model_body.summary(print_fn=log_model_summary)
                    with open(train_annotation_path, encoding='utf-8') as f:
                        train_lines = f.readlines()
                    with open(val_annotation_path, encoding='utf-8') as f:
                        val_lines   = f.readlines()
                    num_train   = len(train_lines)
                    num_val     = len(val_lines)

                    save_dir    = "logs/%s_%s" %(neck, backbone)

                    show_config(
                        classes_path = classes_path, anchors_path = anchors_path, anchors_mask = anchors_mask, model_path = model_path, input_shape = input_shape, \
                        Init_Epoch = Init_Epoch, UnFreeze_Epoch = UnFreeze_Epoch, Unfreeze_batch_size = Unfreeze_batch_size, Freeze_Train = Freeze_Train, \
                        Init_lr = Init_lr, Min_lr = Min_lr, optimizer_type = optimizer_type, momentum = momentum, lr_decay_type = lr_decay_type, \
                        save_period = save_period, save_dir = save_dir, num_workers = num_workers, num_train = num_train, num_val = num_val
                    )


                    wanted_step = 5e4 if optimizer_type == "sgd" else 1.5e4
                    total_step  = num_train // Unfreeze_batch_size * UnFreeze_Epoch
                    if total_step <= wanted_step:
                        if num_train // Unfreeze_batch_size == 0:
                            raise ValueError('Dataset is too small, please extend it.')
                        wanted_epoch = wanted_step // (num_train // Unfreeze_batch_size) + 1

                    if True:

                        batch_size  =  Unfreeze_batch_size

                        nbs             = 64
                        lr_limit_max    = 1e-3 if optimizer_type == 'adam' else 5e-2
                        lr_limit_min    = 3e-4 if optimizer_type == 'adam' else 5e-4
                        Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
                        Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

                        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

                        epoch_step      = num_train // batch_size
                        epoch_step_val  = num_val // batch_size

                        if epoch_step == 0 or epoch_step_val == 0:
                            raise ValueError('Dataset is too small, please extend it.')
                        if neck =='yolov3':
                            train_dataloader    = YoloDatasets(train_lines, input_shape, anchors, batch_size, num_classes, anchors_mask, train=True)
                            val_dataloader      = YoloDatasets(val_lines, input_shape, anchors, batch_size, num_classes, anchors_mask, train=False)
                        elif neck =='yolox' or neck == 'yoloxg':
                            train_dataloader    = YoloDatasets(train_lines, input_shape, batch_size, num_classes, Init_Epoch, UnFreeze_Epoch, \
                                                                mosaic=mosaic, mixup=mixup, mosaic_prob=mosaic_prob, mixup_prob=mixup_prob, train=True, special_aug_ratio=special_aug_ratio)
                            val_dataloader      = YoloDatasets(val_lines, input_shape, batch_size, num_classes, Init_Epoch, UnFreeze_Epoch, \
                                                                mosaic=False, mixup=False, mosaic_prob=0, mixup_prob=0, train=False, special_aug_ratio=0)
                        else:
                            train_dataloader    = YoloDatasets(train_lines, input_shape, anchors, batch_size, num_classes, anchors_mask, Init_Epoch, UnFreeze_Epoch, \
                                                                mosaic=mosaic, mixup=mixup, mosaic_prob=mosaic_prob, mixup_prob=mixup_prob, train=True, special_aug_ratio=special_aug_ratio)
                            val_dataloader      = YoloDatasets(val_lines, input_shape, anchors, batch_size, num_classes, anchors_mask, Init_Epoch, UnFreeze_Epoch, \
                                                                mosaic=False, mixup=False, mosaic_prob=0, mixup_prob=0, train=False, special_aug_ratio=0)

                        optimizer = {
                            'adam'  : Adam(lr = Init_lr, beta_1 = momentum),
                            'sgd'   : SGD(lr = Init_lr, momentum = momentum, nesterov=True)
                        }[optimizer_type]

                        start_epoch = Init_Epoch
                        end_epoch   = UnFreeze_Epoch

                        if ngpus_per_node > 1:
                            with strategy.scope():
                                model.compile(optimizer = optimizer, loss={'yolo_loss': lambda y_true, y_pred: y_pred})
                        else:
                            model.compile(optimizer = optimizer, loss={'yolo_loss': lambda y_true, y_pred: y_pred})

                        time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
                        log_dir         = os.path.join(save_dir, "loss_" + str(time_str))
                        logging         = TensorBoard(log_dir)
                        loss_history    = LossHistory(log_dir)

                        #loss_history.save(os.path.join(save_dir, "ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.txt"))
                        # checkpoint      = ModelCheckpoint(os.path.join(save_dir, "ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5"),
                        #                         monitor = 'val_loss', save_weights_only = True, save_best_only = False, period = save_period)

                        checkpoint_last = ModelCheckpoint(os.path.join(save_dir, "last_epoch_weights.h5"),
                                                monitor = 'val_loss', save_weights_only = True, save_best_only = False, period = 1)
                        checkpoint_best = ModelCheckpoint(os.path.join(save_dir, "best_epoch_weights.h5"),
                                                monitor = 'val_loss', save_weights_only = True, save_best_only = True, period = 1)
                        early_stopping  = EarlyStopping(monitor='val_loss', min_delta = 0, patience = 10, verbose = 1)
                        lr_scheduler    = LearningRateScheduler(lr_scheduler_func, verbose = 1)
                        if neck=='yolox':
                            eval_callback   = EvalCallback(model_body, input_shape, class_names, num_classes, val_lines, log_dir, \
                                                            eval_flag=eval_flag, period=eval_period)
                        else:
                            eval_callback   = EvalCallback(model_body, input_shape, anchors, anchors_mask, class_names, num_classes, val_lines, log_dir, \
                                                            eval_flag=eval_flag, period=eval_period)
                        callbacks       = [logging, loss_history, checkpoint_last, checkpoint_best, lr_scheduler, eval_callback]

                        if start_epoch < end_epoch:
                            print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
                            model.fit(
                                x                   = train_dataloader,
                                steps_per_epoch     = epoch_step,
                                validation_data     = val_dataloader,
                                validation_steps    = epoch_step_val,
                                epochs              = end_epoch,
                                initial_epoch       = start_epoch,
                                use_multiprocessing = True if num_workers > 1 else False,
                                workers             = num_workers,
                                callbacks           = callbacks
                            )
                    #-----------------------------------------------------------#
                    #  Evaluation! Put different results into different floders.
                    #-----------------------------------------------------------#
                    map,fps = get_mapfps(neck,backbone,imagesize)
                    model_size = os.path.getsize('logs/%s_%s/best_epoch_weights.h5' % (neck, backbone))
                    model_name = '%s_%s'%(neck, backbone)
                    parameter = get_params()

                    list_file = open('static/evalu_parames.txt', 'a', encoding='utf-8')
                    list_file.write('{},'.format(model_name))
                    list_file.write('mAP:{}%,'.format(map*100))
                    list_file.write('flops:%sG,' % (flops))
                    list_file.write('model_size:%sMB,' % (model_size/1e6))
                    list_file.write('FPS:%s,' % fps)
                    list_file.write('Parameters:%sM,' % (parameter/1e6))
                    list_file.write('\n')

                    num_backbone += 1
                    num += 1
                    list_file.close()
                    print('{} +'.format(neck),'{} done!'.format(backbone))
            num_neck += 1
    shutil.copy('static/evalu_parames.txt', 'train_results/auto_%s_%s_%s.txt'%(UnFreeze_Epoch, Unfreeze_batch_size, imagesize))
    shutil.copy('static/train.txt', 'train_results/auto_train_%s_%s_%s.txt'%(UnFreeze_Epoch, Unfreeze_batch_size, imagesize))
    print('There are {} necks'.format(num_neck))
    print('There are {} backbones'.format(num_backbone))
    print('There are {} models'.format(num))
    print('Done!')
