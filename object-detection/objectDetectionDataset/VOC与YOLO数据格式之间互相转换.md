# VOC与YOLO数据格式之间互相转换

## 标注工具

使用开源工具LabelImg对图片进行标注，导出的数据集格式为VOC，待数据标注完成后，可以看到文件夹下面这样子的，标注文件XML和图片混在一起

![yolo voc](https://image.xugaoxiang.com/imgs/2021/06/7d7457bef11ae3ba.png)

## 自制VOC数据集

首先，按照VOC2007的数据集格式要求，分别创建文件`VOCdevkit`、`VOC2007`、`Annotations`、`ImageSets`、`Main`和`JPEGImages`，它们的层级结构如下所示

```
└─VOCdevkit
    └─VOC2007
        ├─Annotations
        ├─ImageSets
        │  └─Main
        └─JPEGImages
```

其中，`Annotations`用来存放`xml`标注文件，`JPEGImages`用来存放图片文件，而`ImageSets/Main`存放几个`txt`文本文件，文件的内容是训练集、验证集和测试集中图片的名称(去掉扩展名)，这几个文本文件是需要我们自己生成的，后面会讲到。

接下来，将`images`文件夹中的图片文件拷贝到`JPEGImages`文件夹中，将`images`文件中的`xml`标注文件拷贝到`Annotations`文件夹中

接下来新建一个脚本，把它放在`VOCdevkit/VOC2007`文件夹下，命名为`test.py`

```
─VOCdevkit
    └─VOC2007
        │  test.py
        │
        ├─Annotations
        ├─ImageSets
        │  └─Main
        └─JPEGImages
```

脚本的内容如下：

```
# -*- coding: utf-8 -*-
# @Time    : 2021/6/30 18:24
# @Author  : zhongls
import os
import random

#训练集和验证集的比例分配
trainval_percent = 0.1
train_percent = 0.9

#标注文件的路径
xmlfilepath = "Annotations"

# 生成的txt文件存放路径
txtsavepath = 'ImageSets\Main'
total_xml = os.listdir(xmlfilepath)

num = len(total_xml)
list = range(num)
tv = int(num * trainval_percent)
tr = int(tv * train_percent)
trainval = random.sample(list, tv)
train = random.sample(trainval, tr)

ftrainval = open('ImageSets/Main/trainval.txt', 'w')
ftest = open('ImageSets/Main/test.txt', 'w')
ftrain = open('ImageSets/Main/train.txt', 'w')
fval = open('ImageSets/Main/val.txt', 'w')

for i in list:
    name = total_xml[i][:-4] + '\n'
    if i in trainval:
        ftrainval.write(name)
        if i in train:
            ftest.write(name)
        else:
            fval.write(name)
    else:
        ftrain.write(name)

ftrainval.close()
ftrain.close()
fval.close()
ftest.close()
```

然后，进入到目录`VOCdevkit/VOC2007`，执行这个脚本，结束后，在`ImageSets/Main`下生成了4个`txt`文件

```
├─ImageSets
│  └─Main
│          test.txt
│          train.txt
│          trainval.txt
│          val.txt
│
└─JPEGImages
```

这4个文件的格式都是一样的，文件的内容是对应图片名称去掉扩展名(与`xml`标注文件去掉`.xml`一致)的结果

![yolo voc](https://image.xugaoxiang.com/imgs/2021/06/79c440e489e954f6.png)

有了上面的这些数据准备，最会我们以YOLO中的v3/v4版本为例，看看数据集和训练配置文件是如何结合起来的？

这里，我们下载一个来自`yolo`官方的脚本文件 https://pjreddie.com/media/files/voc_label.py，把`url`贴到浏览器中即可下载

代码比较简单，就是将需要训练、验证、测试的图片绝对路径写到对应的`txt`文件中

```
# -*- coding: utf-8 -*-
# @Time    : 2021/6/30 18:29
# @Author  : zhongls
import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join

# 原始脚本中包含了VOC2012，这里，我们把它删除
# sets=[('2012', 'train'), ('2012', 'val'), ('2007', 'train'), ('2007', 'val'), ('2007', 'test')]
sets=[('2007', 'train'), ('2007', 'val'), ('2007', 'test')]

# classes也需要根据自己的实际情况修改
# classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
classes = ["hat"]

def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def convert_annotation(year, image_id):
    in_file = open('VOCdevkit/VOC%s/Annotations/%s.xml'%(year, image_id))
    out_file = open('VOCdevkit/VOC%s/labels/%s.txt'%(year, image_id), 'w')
    tree=ET.parse(in_file)
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
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

wd = getcwd()

for year, image_set in sets:
    if not os.path.exists('VOCdevkit/VOC%s/labels/'%(year)):
        os.makedirs('VOCdevkit/VOC%s/labels/'%(year))
    image_ids = open('VOCdevkit/VOC%s/ImageSets/Main/%s.txt'%(year, image_set)).read().strip().split()
    list_file = open('%s_%s.txt'%(year, image_set), 'w')
    for image_id in image_ids:
        list_file.write('%s/VOCdevkit/VOC%s/JPEGImages/%s.jpg\n'%(wd, year, image_id))
        convert_annotation(year, image_id)
    list_file.close()
```

执行上述脚本后，在`VOCdevkit`同级目录就会生成`2007_train.txt`、`2007_val.txt`、`2007_test.txt`。

![yolo voc](https://image.xugaoxiang.com/imgs/2021/06/2beef7454dbbeac4.png)

到这里，自制的`VOC2007`数据集就已经准备好了。对应到`darknet`中的配置文件`cfg/voc.data`就可以这么写

```
classes= 1
train  = 2007_train.txt
valid  = 2007_val.txt
names = data/voc.names
backup = backup/
```

## 转换成YOLO数据格式

首先说明一下，前面提到的标注工具`labelImg`可以导出`YOLO`的数据格式。但是如果你拿到的是一份标注格式为`xml`的数据，那就需要进行转换了。拿上面我们自己标注的例子来说

将所有图片存放在`images`文件夹，`xml`标注文件放在`Annotations`文件夹，然后创建一个文件夹`labels`

```
├─Annotations
├─images
└─labels
```

下面准备转换脚本`voc2yolo.py`，部分注释写在代码里

```
import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join

# 根据自己情况修改
classes = ["hat"]

def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

def convert_annotation(image_id):

    if not os.path.exists('Annotations/%s.xml' % (image_id)):
        return

    in_file = open('annotations/%s.xml' % (image_id))

    out_file = open('labels/%s.txt' % (image_id), 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        cls = obj.find('name').text
        if cls not in classes:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

for image in os.listdir('images'):
    # 这里需要根据你的图片情况进行对应修改。比如图片名称是123.456.jpg，这里就会出错了。一般来讲，如果图片格式固定，如全都是jpg，那就image_id=image[:-4]处理就好了。总之，情况比较多，自己看着办，哈哈！
    image_id = image.split('.')[0]
    convert_annotation(image_id)
```

执行上述脚本后，`labels`文件夹就会生成`txt`格式的标注文件了

大家都知道，`yolov5`训练时使用的数据集结构是这样的

```
├─test
│  ├─images
│  └─labels
├─train
│  ├─images
│  └─labels
└─valid
    ├─images
    └─labels
```

因此，我们还需要将图片文件和对应的`txt`标签文件再进行一次划分，首先创建外层的`train`、`valid`、`test`文件夹，然后在每个文件夹底下都分别创建`images`和`labels`文件夹

接下来，可以使用下面的脚本，将图片和标签文件按照比例进行划分

```
import os
import shutil
import random

# 训练集、验证集和测试集的比例分配
test_percent = 0.1
valid_percent = 0.2
train_percent = 0.7

# 标注文件的路径
image_path = 'images'
label_path = 'labels'

images_files_list = os.listdir(image_path)
labels_files_list = os.listdir(label_path)
print('images files: {}'.format(images_files_list))
print('labels files: {}'.format(labels_files_list))
total_num = len(images_files_list)
print('total_num: {}'.format(total_num))

test_num = int(total_num * test_percent)
valid_num = int(total_num * valid_percent)
train_num = int(total_num * train_percent)

# 对应文件的索引
test_image_index = random.sample(range(total_num), test_num)
valid_image_index = random.sample(range(total_num), valid_num) 
train_image_index = random.sample(range(total_num), train_num)

for i in range(total_num):
    print('src image: {}, i={}'.format(images_files_list[i], i))
    if i in test_image_index:
        # 将图片和标签文件拷贝到对应文件夹下
        shutil.copyfile('images/{}'.format(images_files_list[i]), 'test/images/{}'.format(images_files_list[i]))
        shutil.copyfile('labels/{}'.format(labels_files_list[i]), 'test/labels/{}'.format(labels_files_list[i]))
    elif i in valid_image_index:
        shutil.copyfile('images/{}'.format(images_files_list[i]), 'valid/images/{}'.format(images_files_list[i]))
        shutil.copyfile('labels/{}'.format(labels_files_list[i]), 'valid/labels/{}'.format(labels_files_list[i]))
    else:
        shutil.copyfile('images/{}'.format(images_files_list[i]), 'train/images/{}'.format(images_files_list[i]))
        shutil.copyfile('labels/{}'.format(labels_files_list[i]), 'train/labels/{}'.format(labels_files_list[i]))
```

执行代码后，可以看到类似文件层级结构

```
─test
│  ├─images
│  │      1234565343231.jpg
│  │      1559035146628.jpg
│  │      2019032210151.jpg
│  │
│  └─labels
│          1234565343231.txt
│          1559035146628.txt
│          2019032210151.txt
│
├─train
│  ├─images
│  │      1213211.jpg
│  │      12i4u33112.jpg
│  │      1559092537114.jpg
│  │
│  └─labels
│          1213211.txt
│          12i4u33112.txt
│          1559092537114.txt
│
└─valid
    ├─images
    │      120131247621.jpg
    │      124iuy311.jpg
    │      1559093141383.jpg
    │
    └─labels
            120131247621.txt
            124iuy311.txt
            1559093141383.txt
```

至此，数据集就真正准备好了。

## yolo转voc

如果拿到了`txt`的标注，但是需要使用`VOC`，也需要进行转换。看下面这个脚本，注释写在代码中

```
import os
import xml.etree.ElementTree as ET
from PIL import Image
import numpy as np

# 图片文件夹，后面的/不能省
img_path = 'images/'

# txt文件夹，后面的/不能省
labels_path = 'labels/'

# xml存放的文件夹，后面的/不能省
annotations_path = 'Annotations/'

labels = os.listdir(labels_path)

# 类别
classes = ["hat"]

# 图片的高度、宽度、深度
sh = sw = sd = 0

def write_xml(imgname, sw, sh, sd, filepath, labeldicts):
    '''
    imgname: 没有扩展名的图片名称
    '''

    # 创建Annotation根节点
    root = ET.Element('Annotation')

    # 创建filename子节点，无扩展名                 
    ET.SubElement(root, 'filename').text = str(imgname)        

    # 创建size子节点 
    sizes = ET.SubElement(root,'size')                                      
    ET.SubElement(sizes, 'width').text = str(sw)
    ET.SubElement(sizes, 'height').text = str(sh)
    ET.SubElement(sizes, 'depth').text = str(sd) 

    for labeldict in labeldicts:
        objects = ET.SubElement(root, 'object')                 
        ET.SubElement(objects, 'name').text = labeldict['name']
        ET.SubElement(objects, 'pose').text = 'Unspecified'
        ET.SubElement(objects, 'truncated').text = '0'
        ET.SubElement(objects, 'difficult').text = '0'
        bndbox = ET.SubElement(objects,'bndbox')
        ET.SubElement(bndbox, 'xmin').text = str(int(labeldict['xmin']))
        ET.SubElement(bndbox, 'ymin').text = str(int(labeldict['ymin']))
        ET.SubElement(bndbox, 'xmax').text = str(int(labeldict['xmax']))
        ET.SubElement(bndbox, 'ymax').text = str(int(labeldict['ymax']))
    tree = ET.ElementTree(root)
    tree.write(filepath, encoding='utf-8')

for label in labels:
    with open(labels_path + label, 'r') as f:
        img_id = os.path.splitext(label)[0]
        contents = f.readlines()
        labeldicts = []
        for content in contents:
            # 这里要看你的图片格式了，我这里是jpg，注意修改
            img = np.array(Image.open(img_path + label.strip('.txt') + '.jpg'))

            # 图片的高度和宽度
            sh, sw, sd = img.shape[0], img.shape[1], img.shape[2]
            content = content.strip('\n').split()
            x = float(content[1])*sw
            y = float(content[2])*sh
            w = float(content[3])*sw
            h = float(content[4])*sh

            # 坐标的转换，x_center y_center width height -> xmin ymin xmax ymax
            new_dict = {'name': classes[int(content[0])],
                        'difficult': '0',
                        'xmin': x+1-w/2,                     
                        'ymin': y+1-h/2,
                        'xmax': x+1+w/2,
                        'ymax': y+1+h/2
                        }
            labeldicts.append(new_dict)
        write_xml(img_id, sw, sh, sd, annotations_path + label.strip('.txt') + '.xml', labeldicts)
```

执行上述脚本，就可以在`Annotations`看到转换后的`xml`文件了。后面的`VOC`数据集操作请参考文中的第二部分。