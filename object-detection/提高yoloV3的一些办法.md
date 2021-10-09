随手记，看到别人写的，觉得蛮有道理的，有时间去实现看看，是否有用

1.random=1可以设置适应多分辨率

2.提升分辨率：416--> 608等必须是32倍数

3.重新计算你的数据集的anchor:(注意设置的时候计算问题)

darknet.exe detector calc_anchors data/obj.data -num_of_clusters 9 -width 416 -height 416

检查数据集通过https://github.com/AlexeyAB/Yolo_mark

4.数据集最好每个类有2000张图片，至少需要迭代2000*类的个数

5.数据集最好有没有标注的对象，即负样本，对应空的txt文件，最好有多少样本就设计多少负样本。

6.对于一张图有很多个样本的情况，使用max=200属性(yolo层或者region层)

7.for training for small objects - set layers = -1, 11 instead of https://github.com/AlexeyAB/darknet/blob/6390a5a2ab61a0bdf6f1a9a6b4a739c16b36e0d7/cfg/yolov3.cfg#L720 and set stride=4 instead of https://github.com/AlexeyAB/darknet/blob/6390a5a2ab61a0bdf6f1a9a6b4a739c16b36e0d7/cfg/yolov3.cfg#L717

8.训练数据需要满足以下条件：

train_network_width * train_obj_width / train_image_width ~= detection_network_width * detection_obj_width / detection_image_width
train_network_height * train_obj_height / train_image_height ~= detection_network_height * detection_obj_height / detection_image_height
9.为了加速训练，可以做fine-tuning而不是从头开始训练，设置stopbackward=1在网络的结束部分（以####作为分割）

10.在训练完以后，进行目标检测的时候，可以提高网络的分辨率，以便刚好检测小目标。

11.不需要重新训练，需要使用原先低分辨率的权重，测用更高分辨率。
12.为了得到更高的检测效果，可以提升分辨率至608*608甚至832*832