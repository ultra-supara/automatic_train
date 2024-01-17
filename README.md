# AI model automatic generation software for 20-dishs

<!-- <h1 align="center">
  <a href="https://github.com/dec0dOS/amazing-github-template">
    <img src="{{cookiecutter.repo_slug}}/docs/images/logo.svg" alt="Logo" width="125" height="125">
  </a>
</h1> -->

<div align="center">
<br />

[![license](https://img.shields.io/github/license/dec0dOS/amazing-github-template.svg?style=flat-square)](LICENSE)
</div>

<details open="open">
<summary>Table of Contents</summary>

- [About](#about)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Usage](#usage)
    - [automatic_train kick off](#automatic_train-kickoff)
    - [Variables reference](#variables-reference)
    - [sample_training](#sample_training)
    - [results](#results)
    - [Demonstration](#demonstration)
- [License](#license)

</details>

---

## About

<table>
<tr>
<td>

It is the best algorithm to automate the process of building the optimal model for AI image recognition. The data is also returned to the web as an example of an application using the resources collected there, allowing for machine learning from the web.The system can provide "generated and compressed compact AI models," "weight information," and "classified images spit out by the generated models. It is also possible to retrieve desired information such as mAP (recognition rate) and Flops (computational cost) that are generated in the process of generation.

1. Architecture

<img width="1239" alt="スクリーンショット 0006-01-17 11 18 56" src="https://github.com/ultra-supara/automatic_train/assets/67861004/cf99eef6-2806-40af-8edf-ad0bb0f7b052">

This is an algorithm that automatically learns and determines which combination of backbone and neck & head is optimal for the collected annotated image data, and automatically outputs the optimal AI model.

2. Compression of generated AI models

The second is compression technology for AI models. Recent developments in hardware devices have enabled the storage of huge amounts of information and large-scale computation, which have brought about an evolution in AI technology. However, many existing AI models contain a lot of redundant computations that do not fully consider the use of resources in the pursuit of recognition accuracy.

The AI models that have been generated at this stage of development, while not having significant accuracy problems (85-95% of conventional methods), are large and far from "compact. In this case, the key is to "compress the software to a compact size while maintaining accuracy," which is a difficult problem to solve, and indeed, we are facing challenges in this area.

However, once this issue is overcome, the technology can be used in a wider range of applications because it can be realized on smaller devices at lower cost.

Key features of **automatic_train**:

- img : save image file of 20 dish samples (private)
- templates : html file for web application
- utils : mishmash of resources used in learning
- yolomodels : net... yolo...many files describe the heart of yolo's network architecture, which leverages resources in utils for learning.
- static : It contains pre-annotated and post-annotated images.
           In addition, there are txt files of the training results and the deep learning transition results from the automatic training.

`model_data/dish_classes.txt` : 20 dishs detection label is here

```yaml
Chopsticks-cover
Chopsticks-one
Chopsticks-two
Coffee
Coffee-cup
Coffee-dish
Cup
Fish-dish
Paper
Rice-bowl
Soup-bowl
Spoon
Square-bowl
Tea-cup
Tea-dish
Towel
Towel-dish
Waster-paper
Water-cup
Wine-cup
```

<details open>
<summary>Additional info</summary>
<br>

Image recognition includes techniques for visual inspection for defects, defects, dirt, etc., and object detection. In the past, general image processing was used for image recognition in appearance inspection and object detection. However, conventional image processing methods have difficulties in detecting defects in object contours and handling image recognition due to changes in light.

This software was created to overcome these challenges. When an optimal model can be output as a result of combinatorial learning on an input image, it shows performance far beyond human judgment.

</details>

</td>
</tr>
</table>

## Getting Started

### Prerequisites

The easiest way to install automatic_train is by running:

```sh
git clone https://github.com/ultra-supara/automatic_train.git
```

### Usage

#### automatic_train kick off

After installing automatic_train , you need to do is to run the following command:

```sh
pip install -r requirements.txt
```

#### Variables reference

Please note that entered values are case-sensitive.
Default values are provided as an example to help you figure out what should be entered.

> On manual setup, you need to replace only values written in **uppercase**.

| Name                       | Tips               | Description                                                                 |
| -------------------------- | ------------------ | --------------------------------------------------------------------------- |
| PROJECT_NAME               | automatic_train    | My project name                                                      |
| GITHUB_USERNAME            | ultra-supara       | My GitHub username                                                          |
| FULL_NAME                  | sada atsushi       | My realname                                                                 |
| OPEN_SOURCE_LICENSE        | MIT license        | Full OSS license                                                            |
| app.py                     | python3 app.py     | url [Home](http://127.0.0.1:8001/)                                          |
| first_option_auto_train    | simply option      | url [autotrain](http://127.0.0.1:8001/autotrain/)                           |
| second_option_auto_train   | more detailed opt  | url [manualtrain](http://127.0.0.1:8001/manualtrain/)                       |
| upload_place               | ihpc               | upload.php [ihpc server](http://www.ihpc.se.ritsumei.ac.jp/iot/auto-test/)  |

launch Flask app

```sh
python3 app.py
```

then , you may see this url [Home](http://127.0.0.1:8001/)

**The deep learning part can all be operated from the browser side!**
We are currently preparing a Docker environment in which it can run.

#### sample_training

This is an example of training with [autotrain](http://127.0.0.1:8001/autotrain/) , `One-Key Training` (a mechanism that allows machine learning from a browser without the need to adjust parameters).

<img width="1373" alt="スクリーンショット 0005-05-08 13 57 59" src="https://user-images.githubusercontent.com/67861004/236739559-b9048bca-9bf1-4bfb-874b-0da206a7bfc9.png">

The training conditions are in `train_onekey.py`
```py
# 学習パラメタの設定
if __name__ == "__main__":
    imagesize = 32  #N*32(N=(1,20))
    epochs = 1
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
```

#### results
This is an example of the results (`static/train.txt`) when the combination of yolov7_v5CSPdarknet is turned around training with an epochs count of 50.

```cpp
logs/yolov7_v5CSPdarknet/best_epoch_weights.h5 model, anchors, and classes loaded.
Configurations: ** each part of the network **
----------------------------------------------------------------------
|                     keys |                                   values|
----------------------------------------------------------------------
|                     neck |                                         |
|                 backbone |                                         |
|             classes_path |               model_data/voc_classes.txt|
|             anchors_path |              model_data/yolo_anchors.txt|
|             anchors_mask |        [[6, 7, 8], [3, 4, 5], [0, 1, 2]]|
|              input_shape |                                       []|
|               confidence |                                      0.5|
|                  nms_iou |                                      0.3|
|                max_boxes |                                      100|
|          letterbox_image |                                     True|
----------------------------------------------------------------------
Load model done.
Get predict result.
Get predict result done.
Get ground truth result.
Get ground truth result done.
Get map.
...
yolov7 + v5CSPdarknet done!
There are 6 backbones.
Current training model: yolov7 + xCSPdarknet!
Evaluation parameter: mAP
Total GFLOPs: 0.284G

Configurations: ** each parameters **
----------------------------------------------------------------------
|                     keys |                                   values|
----------------------------------------------------------------------
|             classes_path |               model_data/voc_classes.txt|
|             anchors_path |              model_data/yolo_anchors.txt|
|             anchors_mask |        [[6, 7, 8], [3, 4, 5], [0, 1, 2]]|
|               model_path |                                         |
|              input_shape |                                 [32, 32]|
|               Init_Epoch |                                        0|
|           UnFreeze_Epoch |                                       50|
|      Unfreeze_batch_size |                                        2|
|             Freeze_Train |                                    False|
|                  Init_lr |                                    0.001|
|                   Min_lr |                                    1e-05|
|           optimizer_type |                                     adam|
|                 momentum |                                    0.937|
|            lr_decay_type |                                      cos|
|              save_period |                                       10|
|                 save_dir |                  logs/yolov7_xCSPdarknet|
|              num_workers |                                        1|
|                num_train |                                      409|
|                  num_val |                                       46|
----------------------------------------------------------------------
Train on 409 samples, val on 46 samples, with batch size 2.
----------------------------------------------------------------------

yolov7 + v5CSPdarknet done!
There are 6 backbones.
Current training model: yolov7 + xCSPdarknet!
```

This part shows the yolov7 + v5CSPdarknet combination has been trained, and the next part shows the yolov7 + xCSPdarknet being trained in succession. These can be done fully automatically. We have succeeded in automating the recombination process that was previously done manually by humans.

detection best model : `static/evalu_parames.txt`

```yaml
yolov7_v5CSPdarknet,mAP:100.0%,flops:0.284323918G,model_size:204.553464MB,FPS:20.251316360863697,Parameters:50.902497M,
yolov7_xCSPdarknet,mAP:95.2%,flops:0.284323918G,model_size:204.579944MB,FPS:17.79088361770951,Parameters:50.902497M,
yolov7_v3darknet,mAP:92.4%,flops:0.385888334G,model_size:258.473392MB,FPS:17.01340010345215,Parameters:64.411713M,
yolov7_v7backbone,mAP:93.2%,flops:0.262721614G,model_size:150.126088MB,FPS:25.20116813428497,Parameters:37.346753M,
yolov7_v4CSPdarknet53,mAP:93.2%,flops:0.29979451G,model_size:202.75124MB,FPS:18.02773895714467,Parameters:50.443585M,
yolov5_v5CSPdarknet,mAP:87.2%,flops:0.232378318G,model_size:154.303304MB,FPS:24.313955865774663,Parameters:38.357737M,
yolov5_xCSPdarknet,mAP:85.4%,flops:0.232378318G,model_size:154.306808MB,FPS:29.451885644243497,Parameters:38.357737M,
yolov5_v3darknet,mAP:86.8%,flops:0.333942734G,model_size:208.223072MB,FPS:20.756551659258147,Parameters:51.866953M,
yolov5_v7backbone,mAP:86.3%,flops:0.211431374G,model_size:100.07604MB,FPS:45.17792878841556,Parameters:24.853193M,
yolov5_v4CSPdarknet53,mAP:88.0%,flops:0.24784891G,model_size:152.497128MB,FPS:28.897781129573318,Parameters:37.898825M,
yolov3_v5CSPdarknet,mAP:80.0%,flops:0.28694587G,model_size:193.377224MB,FPS:21.945551351170344,Parameters:48.169441M,
yolov3_xCSPdarknet,mAP:74.6%,flops:0.28694587G,model_size:193.385488MB,FPS:21.715122896668202,Parameters:48.169441M,
yolov3_v3darknet,mAP:77.4%,flops:0.388510286G,model_size:247.291248MB,FPS:17.55582086893424,Parameters:61.678657M,
yolov3_v7backbone,mAP:72.3%,flops:0.265343566G,model_size:138.944872MB,FPS:29.82171052299465,Parameters:34.613697M,
yolov3_v4CSPdarknet53,mAP:75.9%,flops:0.302416462G,model_size:191.576568MB,FPS:21.516757871989906,Parameters:47.710529M,
![image](https://github.com/ultra-supara/automatic_train/assets/67861004/c51593b0-47ae-4daa-b73d-173fe02e6a15)

```

#### Demonstration

In this video , first_option_auto_train [automation_demonstraion](https://www.youtube.com/watch?v=eW_ryVj-i5g)

## License

This project is licensed under the **MIT license**. Feel free to edit and distribute this template as you like.

See [LICENSE](LICENSE) for more information.
