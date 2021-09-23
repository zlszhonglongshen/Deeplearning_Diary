# yolact-训练自己的数据集

## yolact简介

```
https://blog.csdn.net/jiaoyangwm/article/details/89176767
```



## 安装labelme

```
pip3 install labelme

labelme
```

打开后放入图片并进行标注，如：

![image-20210324195647022](C:\Users\zhong\AppData\Roaming\Typora\typora-user-images\image-20210324195647022.png)

寻找合适的图片并进行一一标注，保存后会生成对应的json文件。

## 下载代码并安装必要的程序包

```
git clone https://github.com/x0rzkov/yolact-mini-docker.git
```

## 模型训练

### 更改../data/config.py文件

1.更改custom相关信息

```
CUSTOM_CLASSES = ('smoke',)  #如果只有一样物件需要加上,

CUSTOM_LABEL_MAP = {1: 1}

custom_dataset = Config({'name': 'Custom dataset',
                         'train_images': '/home/jiang/yolact-mini-docker/results/images/output/',  # No need to add 'JPEGImages/'.
                         'train_info': '/home/jiang/yolact-mini-docker/results/images/output/annotations.json',
                         'valid_images': '/home/jiang/yolact-mini-docker/results/images/output/',
                         'valid_info': '/home/jiang/yolact-mini-docker/results/images/output/annotations.json',
                         'class_names': CUSTOM_CLASSES})
#修改路径                         
 
res101_custom_config = res101_coco_config.copy({
    'name': 'res101_custom',
    'dataset': custom_dataset,
    'num_classes': len(custom_dataset.class_names) + 1,
    'batch_size': 8,
    'img_size': 550,  # image size
    'max_iter': 800000,
    'backbone': resnet101_backbone,
    # During training, first compute the maximum gt IoU for each prior.
    # Then, for priors whose maximum IoU is over the positive threshold, marked as positive.
    # For priors whose maximum IoU is less than the negative threshold, marked as negative.
    # The rest are neutral ones and are not used to calculate the loss.
    'pos_iou_thre': 0.5,
    'neg_iou_thre': 0.4,
    # If less than 1, anchors treated as a negative that have a crowd iou over this threshold with
    # the crowd boxes will be treated as a neutral.
    'crowd_iou_threshold': 0.7,
    'conf_alpha': 1,
    'bbox_alpha': 1.5,
    'mask_alpha': 6.125,
    # Learning rate
    'lr_steps': (280000, 600000, 700000, 750000),
    #'lr_steps': (280000,360000,400000),
    'lr': 1e-3,
    'momentum': 0.9,
    'decay': 5e-4,
    # warm up setting
    'warmup_init': 1e-4,
    'warmup_until': 500,
    # The max number of masks to train for one image.
    'masks_to_train': 100,
    # anchor settings
    'scales': [24, 48, 96, 192, 384],
    'aspect_ratios': [1, 1 / 2, 2],
    'use_square_anchors': True,  # This is for backward compatability with a bug.
    # Whether to train the semantic segmentations branch, this branch is only implemented during training.
    'train_semantic': True,
    'semantic_alpha': 1,
    # postprocess hyperparameters
    'conf_thre': 0.05,
    'nms_thre': 0.5,
    'top_k': 200,
    'max_detections': 100,
    # Freeze the backbone bn layer during training, other additional bn layers after the backbone will not be frozen.
    'freeze_bn': False,
    'label_map': CUSTOM_LABEL_MAP})     
#将res101_custom_config替换成可改动的参数                       

```

### 在该目录下定义一个labels.txt文件

```
__ignore__
_background_
smoke

```

### 在utils目录下执行

```
python3 labelme2coco.py your-image-and-labelme-json-path your-expected-output-folder --labels the-path-of-labels.txt

#样例
python3 labelme2coco.py /home/jiang/yolact-mini-docker/image /home/jiang/yolact-mini-docker/results/images/output --labels /home/jiang/yolact-mini-docker/labels.txt
#注意每次执行这句语句需要清空上次生成的文件夹

```

执行之后会生成一个文件夹并将之前标注过的图片放入，并会生成一个对应的annotations.json文件

### 训练

```
python3 train.py --config=res101_custom_config

```

### 评估

```
python3 eval.py --trained_model=latest_res101_custom_801.pth --max_num=1000#注意命名的pth是刚刚自己生成的.pth文件

```

### 测试

```
python3 detect.py --trained_model=latest_res101_custom_601.pth --image /home/jiang/1#路径为想要检测的图片所在文件夹

```

### 结果展示

![image-20210324200655981](C:\Users\zhong\AppData\Roaming\Typora\typora-user-images\image-20210324200655981.png)

### 参考链接

* https://blog.csdn.net/weixin_44878465/article/details/108149285
* https://github.com/dbolya/yolact

* https://github.com/x0rzkov/yolact-mini-docker
* https://github.com/feiyuhuahuo/Yolact_minimal

