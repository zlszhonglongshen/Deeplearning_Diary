

# YOLOV5-4.0训练自己的数据集

# 环境准备

## python环境

```
python>=3.8.0
```

## pytorch环境

```
pip install torch==1.7.0+cu101 torchvision==0.8.1+cu101 torchaudio===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
```

检查是否安装成功

```
import torch
torch.cuda.is_available()
```

# 下载源码以及安装依赖

```
pip install -r requirements.txt
```

# 数据标注以及数据准备

## 数据标注

这里主要利用labelme软件进行数据标注，关于该软件的使用，可以参考前面的文章；最后我们将图像放在data/images目录下，xml文件放在data/Annotations目录下。

## 数据预处理

### 划分训练集和测试集

```
import os
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--xml_path', default='data/Annotations', type=str, help='input xml label path')
parser.add_argument('--txt_path', default='data/labels', type=str, help='output txt label path')
opt = parser.parse_args()

trainval_percent = 1.0
train_percent = 0.9
xmlfilepath = opt.xml_path
txtsavepath = opt.txt_path
total_xml = os.listdir(xmlfilepath)
if not os.path.exists(txtsavepath):
    os.makedirs(txtsavepath)

num = len(total_xml)
list_index = range(num)
tv = int(num * trainval_percent)
tr = int(tv * train_percent)
trainval = random.sample(list_index, tv)
train = random.sample(trainval, tr)

file_trainval = open(txtsavepath + '/trainval.txt', 'w')
file_test = open(txtsavepath + '/test.txt', 'w')
file_train = open(txtsavepath + '/train.txt', 'w')
file_val = open(txtsavepath + '/val.txt', 'w')

for i in list_index:
    name = total_xml[i][:-4] + '\n'
    if i in trainval:
        file_trainval.write(name)
        if i in train:
            file_train.write(name)
        else:
            file_val.write(name)
    else:
        file_test.write(name)

file_trainval.close()
file_train.close()
file_val.close()
file_test.close()
```

### 将数据集转化为yolo数据格式

```
# -*- coding: utf-8 -*-

import xml.etree.ElementTree as ET
from tqdm import tqdm
import os
from os import getcwd

sets = ['train', 'val', 'test']
classes = ['Helmet','person']


def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return x, y, w, h


def convert_annotation(image_id):
    # try:
        in_file = open('VOCData/Annotations/%s.xml' % (image_id), encoding='utf-8')
        out_file = open('VOCData/labels/%s.txt' % (image_id), 'w', encoding='utf-8')
        tree = ET.parse(in_file)
        root = tree.getroot()
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)
        for obj in root.iter('object'):
            difficult = obj.find('difficult').text
            cls = obj.find('name').text
            if cls not in classes or int(difficult) == 1:
                continue
            cls_id = classes.index(cls)
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
                 float(xmlbox.find('ymax').text))
            b1, b2, b3, b4 = b
            # 标注越界修正
            if b2 > w:
                b2 = w
            if b4 > h:
                b4 = h
            b = (b1, b2, b3, b4)
            bb = convert((w, h), b)
            out_file.write(str(cls_id) + " " +
                           " ".join([str(a) for a in bb]) + '\n')
    # except Exception as e:
    #     print(e, image_id)


wd = getcwd()
for image_set in sets:
    if not os.path.exists('VOCData/labels/'):
        os.makedirs('VOCData/labels/')
    image_ids = open('VOCData/labels/%s.txt' %
                     (image_set)).read().strip().split()
    list_file = open('VOCData/%s.txt' % (image_set), 'w')
    for image_id in tqdm(image_ids):
        list_file.write('VOCData/images/%s.jpg\n' % (image_id))
        convert_annotation(image_id)
    list_file.close()
```

代码运行成功后，在data/labels下生成每个图的txt文件。

### 在data文件夹下创建helmet.yaml文件,并根据自己的具体情况修改如下：

```
# download command/URL (optional)
download: bash data/scripts/get_voc.sh

# train and val data as 1) directory: path/images/, 2) file: path/images.txt, or 3) list: [path1/images/, path2/images/]
train: data/train.txt  # 16551 images
val: data/val.txt  # 4952 images

# number of classes
nc: 2

# class names
names: [ 'helmet','person' ]
```

# 模型训练

## 预训练模型下载

这里我选择的是系列里面最小的模型，yolo5s.pt

## 开始训练

修改models/yolov5s.yaml的类别数：

```
# parameters
nc: 2  # number of classes
depth_multiple: 0.33  # model depth multiple
width_multiple: 0.50  # layer channel multiple
```

### 训练命令行

```
python train.py --img 640 --batch 4 --epoch 300 --data ./data/helmet.yaml --cfg ./models/yolov5s.yaml --weights weights/yolov5s.pt --workers 0
```

