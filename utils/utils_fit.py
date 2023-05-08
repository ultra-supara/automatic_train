'''
このスクリプトは、TensorFlowを使ってYOLOXの物体検出モデルを学習・検証するために使用します。
主要な関数は `fit_one_epoch` で、1つのエポックについてモデルの学習と検証を行います。
トレーニングステップと検証ステップは、それぞれ `get_train_step_fn` と `get_val_step_fn` 関数を使用して定義します。それぞれの損失計算ロジックとともに作成する。

fit_one_epoch`関数を定義します。この関数は、1つのエポックについてモデルのトレーニングと検証を行うものです。この関数は
    - 訓練ステップと検証ステップを設定するために、訓練ステップと検証ステップの関数を呼び出す。
    - 訓練データと検証データをループし、提供されたデータで訓練ステップまたは検証ステップを実行します。
    - 各エポックの終了時に、平均損失と検証損失を計算する。
    - save_period`で指定された条件が満たされた場合、エポック終了時にモデルの重みを保存する。
    - 現在の検証損失が前回の検証損失より小さい場合、最適なモデルの重みを保存する。
    - 最後のモデルの重みを保存する。

このスクリプトを使ってYOLOXモデルを学習するには、モデル、オプティマイザー、入力形状、アンカーなどの必要な引数を与えて、
回したいエポック数だけループで `fit_one_epoch` 関数を呼び出す必要があります。
また、学習プロセスでモデルに与えるトレーニングデータと検証データ（`gen`と`gen_val`）のデータジェネレータを設定する必要があります。
'''

import os

import tensorflow as tf
from yolomodels.yolos.yolox import yolo_loss
from tqdm import tqdm
#------------------------------#
#   bug防止機構
#------------------------------#
def get_train_step_fn(input_shape, anchors, anchors_mask, num_classes, label_smoothing, focal_loss, alpha, gamma, strategy):
    @tf.function
    def train_step(imgs, targets, net, optimizer):
        with tf.GradientTape() as tape:
            #------------------------------#
            #   lossを算出する
            #------------------------------#
            P5_output, P4_output, P3_output = net(imgs, training=True)
            args        = [P5_output, P4_output, P3_output] + targets
            loss_value  = yolo_loss(
                args, input_shape, anchors, anchors_mask, num_classes,
                label_smoothing = label_smoothing,
                balance         = [0.4, 1.0, 4],
                box_ratio       = 0.05,
                obj_ratio       = 5 * (input_shape[0] * input_shape[1]) / (416 ** 2),
                cls_ratio       = 1 * (num_classes / 80),
                focal_loss      = focal_loss,
                focal_loss_ratio= 10,
                alpha           = alpha,
                gamma           = gamma
            )
            #------------------------------#
            #   添加上l2正则化参数
            #------------------------------#
            loss_value  = tf.reduce_sum(net.losses) + loss_value
        grads = tape.gradient(loss_value, net.trainable_variables)
        optimizer.apply_gradients(zip(grads, net.trainable_variables))
        return loss_value

    if strategy == None:
        return train_step
    else:
        #----------------------#
        #   多gpu訓練
        #----------------------#
        @tf.function
        def distributed_train_step(images, targets, net, optimizer):
            per_replica_losses = strategy.run(train_step, args=(images, targets, net, optimizer,))
            return strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses,
                                    axis=None)
        return distributed_train_step

#----------------------#
#   防止bug
#----------------------#
def get_val_step_fn(input_shape, anchors, anchors_mask, num_classes, label_smoothing, focal_loss, alpha, gamma, strategy):
    @tf.function
    def val_step(imgs, targets, net, optimizer):
        #------------------------------#
        #   計算loss
        #------------------------------#
        P5_output, P4_output, P3_output = net(imgs, training=False)
        args        = [P5_output, P4_output, P3_output] + targets
        loss_value  = yolo_loss(
            args, input_shape, anchors, anchors_mask, num_classes,
            label_smoothing = label_smoothing,
            balance         = [0.4, 1.0, 4],
            box_ratio       = 0.05,
            obj_ratio       = 5 * (input_shape[0] * input_shape[1]) / (416 ** 2),
            cls_ratio       = 1 * (num_classes / 80),
            focal_loss      = focal_loss,
            focal_loss_ratio= 10,
            alpha           = alpha,
            gamma           = gamma
        )
        #------------------------------#
        #   L2正則化パラメータを追加します。
        #------------------------------#
        loss_value  = tf.reduce_sum(net.losses) + loss_value
        return loss_value
    if strategy == None:
        return val_step
    else:
        #----------------------#
        #   複数のGPUで検証します。
        #----------------------#
        @tf.function
        def distributed_val_step(images, targets, net, optimizer):
            per_replica_losses = strategy.run(val_step, args=(images, targets, net, optimizer,))
            return strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses,
                                    axis=None)
        return distributed_val_step

def fit_one_epoch(net, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch,
            input_shape, anchors, anchors_mask, num_classes, label_smoothing, focal_loss, alpha, gamma, save_period, save_dir, strategy):
    train_step  = get_train_step_fn(input_shape, anchors, anchors_mask, num_classes, label_smoothing, focal_loss, alpha, gamma, strategy)
    val_step    = get_val_step_fn(input_shape, anchors, anchors_mask, num_classes, label_smoothing, focal_loss, alpha, gamma, strategy)

    loss        = 0
    val_loss    = 0
    print('Start Train')
    with tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_step:
                break
            images, target0, target1, target2 = batch[0], batch[1], batch[2], batch[3]
            targets     = [target0, target1, target2]
            loss_value  = train_step(images, targets, net, optimizer)
            loss        = loss + loss_value

            pbar.set_postfix(**{'total_loss': float(loss) / (iteration + 1),
                                'lr'        : optimizer.lr.numpy()})
            pbar.update(1)
    print('Finish Train')

    print('Start Validation')
    with tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen_val):
            if iteration >= epoch_step_val:
                break
            images, target0, target1, target2 = batch[0], batch[1], batch[2], batch[3]
            targets     = [target0, target1, target2]
            loss_value  = val_step(images, targets, net, optimizer)
            val_loss    = val_loss + loss_value

            pbar.set_postfix(**{'total_loss': float(val_loss) / (iteration + 1)})
            pbar.update(1)
    print('Finish Validation')

    logs = {'loss': loss.numpy() / epoch_step, 'val_loss': val_loss.numpy() / epoch_step_val}
    loss_history.on_epoch_end([], logs)
    eval_callback.on_epoch_end(epoch, logs)
    print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
    print('Total Loss: %.3f || Val Loss: %.3f ' % (loss / epoch_step, val_loss / epoch_step_val))

    #-----------------------------------------------#
    #   重みを保存します。
    #-----------------------------------------------#
    if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
        net.save_weights(os.path.join(save_dir, "ep%03d-loss%.3f-val_loss%.3f.h5" % (epoch + 1, loss / epoch_step, val_loss / epoch_step_val)))

    if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
        print('Save best model to best_epoch_weights.pth')
        net.save_weights(os.path.join(save_dir, "best_epoch_weights.h5"))

    net.save_weights(os.path.join(save_dir, "last_epoch_weights.h5"))
