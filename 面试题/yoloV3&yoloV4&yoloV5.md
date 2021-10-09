# 参考链接

* [一文读懂YOLO V5 与 YOLO V4](https://blog.csdn.net/williamhyin/article/details/107717304)
* [深入浅出Yolo系列之Yolov5核心基础知识完整讲解](https://blog.csdn.net/nan355655600/article/details/107852353?utm_medium=distribute.pc_aggpage_search_result.none-task-blog-2~all~top_click~default-1-107852353.nonecase&utm_term=yolov5%E8%AE%B2%E8%A7%A3&spm=1000.2123.3001.4430)
* [YOLOv5s网络结构详解](https://blog.csdn.net/Augurlee/article/details/108020933)
* [深入浅出Yolo系列之Yolov3&Yolov4&Yolov5核心基础知识完整讲解](https://blog.csdn.net/nan355655600/article/details/106246625)
* [YOLOv3和YOLOv4长篇核心综述（上）](https://www.cnblogs.com/wujianming-110117/p/13023126.html)
* [YOLOv3和YOLOv4长篇核心综述（下）](https://www.cnblogs.com/wujianming-110117/p/13023133.html)
* [深入浅出YOLOV3和YOLOV4](https://blog.csdn.net/l7H9JA4/article/details/106416463)
* [[Yolov3&Yolov4网络结构与源码分析](https://www.cnblogs.com/wujianming-110117/p/13845974.html)](https://www.cnblogs.com/wujianming-110117/p/13845974.html)
* [YOLO系列：YOLOv1,YOLOv2,YOLOv3,YOLOv4,YOLOv5简介](https://www.baidu.com/link?url=OiYHLMn1m-B_yg7hsrWhNkXI7V6TwKnDk1FLld6iD5KerxG7eCy7QrsixwBUQi4a&wd=&eqid=c9dca79d00081daa000000065fa76b59)

* [一文看懂YOLO v3](https://blog.csdn.net/litt1e/article/details/88907542)



## yoloV4与yoloV5

YOLO网络主要由三个组件组成。

1、Backbone：在不同图像细粒度上聚合形成图像特征的卷积神经网络

2、Neck：一系列混合和组成图像特征的网络层，并将图像特征传递到预测层

3、Head：对图像特征进行预测，生成边界框和并预测类别

目标检测的通用框架：

![img](https://imgconvert.csdnimg.cn/aHR0cHM6Ly93aWxsaWFtaHlpbi0xMzAxNDA4NjQ2LmNvcy5hcC1zaGFuZ2hhaS5teXFjbG91ZC5jb20vaW1nLzIwMjAwNzEyMDk1NTU0LnBuZw?x-oss-process=image/format,png)

实际上YOLO V5的模型架构是与V4非常相近的。在下文中，我会从下面几个方面对比YOLO V5和V4，并简要阐述它们各自新技术的特点，对比两者的区别和相似之处，评判两者的性能，并做最后总结。

- **Data Augmentation**
- **Auto Learning Bounding Box Anchors**
- **Backbone**
- **Neck**
- **Head**
- **Network Architecture**
- **Activation Function**
- **Optimization Function**
- **Benchmarks**



### Data Augmentation

图像增强是从现有的训练数据中创建新的训练样本，我们不可能为每一个现实世界场景捕捉一个图像，因此我们需要调整现有的训练数据以推广到其他情况，从而允许模型适应更广泛的情况。无论是V5还是V4，多样化的先进数据增强技术是最大限度的利用数据集，使对象检测框取得性能突破的关键。通过一系列的图像增强技术步骤，可以在不增加推理延时的情况下，提高模型的性能。

yoloV4数据增强



![在这里插入图片描述](https://img-blog.csdnimg.cn/20200731170445153.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dpbGxpYW1oeWlu,size_16,color_FFFFFF,t_70)

yoloV4使用了上图中多种数据增强技术的组合，对于单一突破，除了经典的几何畸变与光照畸变外，还创新的使用了图像遮挡（random erase，cutout，hide and seek，grid mask，mixup）技术，对于多图组合，作者混合使用cutmix和mosaic技术，除此之外，作者还使用了SAT来进行数据增强。

### 图像遮挡

* random Erase：用随机值或训练集的平均像素值替换图像的区域

  ![img](https://imgconvert.csdnimg.cn/aHR0cHM6Ly93aWxsaWFtaHlpbi0xMzAxNDA4NjQ2LmNvcy5hcC1zaGFuZ2hhaS5teXFjbG91ZC5jb20vaW1nLzIwMjAwNzEyMTExMTIwLnBuZw?x-oss-process=image/format,png)

* cutout：仅对CNN第一层的输入使用剪切方块Mask.

  ![img](https://imgconvert.csdnimg.cn/aHR0cHM6Ly93aWxsaWFtaHlpbi0xMzAxNDA4NjQ2LmNvcy5hcC1zaGFuZ2hhaS5teXFjbG91ZC5jb20vaW1nLzIwMjAwNzEyMTExNDExLnBuZw?x-oss-process=image/format,png)

* Hide and Seek:将图像分割成一个由S*S图像补丁组成的网络，根据概率设置随机隐藏一些补丁，从而模型学习整个对象的样子，而不是单独一块，比如不单独依赖动物的脸做识别

  

  ![img](https://imgconvert.csdnimg.cn/aHR0cHM6Ly93aWxsaWFtaHlpbi0xMzAxNDA4NjQ2LmNvcy5hcC1zaGFuZ2hhaS5teXFjbG91ZC5jb20vaW1nLzIwMjAwNzEyMTExNzEzLnBuZw?x-oss-process=image/format,png)

* Grid Mask：将图像的区域隐藏在网格中，作用也是为了让模型学习对象的整个组成部分。

  ![img](https://imgconvert.csdnimg.cn/aHR0cHM6Ly93aWxsaWFtaHlpbi0xMzAxNDA4NjQ2LmNvcy5hcC1zaGFuZ2hhaS5teXFjbG91ZC5jb20vaW1nLzIwMjAwNzEyMTExOTQ5LnBuZw?x-oss-process=image/format,png)

* MixUp：图像对其标签的凸面叠加
* ![img](https://imgconvert.csdnimg.cn/aHR0cHM6Ly93aWxsaWFtaHlpbi0xMzAxNDA4NjQ2LmNvcy5hcC1zaGFuZ2hhaS5teXFjbG91ZC5jb20vaW1nLzIwMjAwNzEyMTEyMDE3LnBuZw?x-oss-process=image/format,png)

### 多图组合

* cutmix：将另外一个图像中的剪切部分粘贴到增强图像，图像的剪切迫使模型学会根据大量的特征进行预测。

  ![img](https://imgconvert.csdnimg.cn/aHR0cHM6Ly93aWxsaWFtaHlpbi0xMzAxNDA4NjQ2LmNvcy5hcC1zaGFuZ2hhaS5teXFjbG91ZC5jb20vaW1nLzIwMjAwNzEyMTEyMzA5LnBuZw?x-oss-process=image/format,png)

* mosaic data augmentation

  在cutmix中我们组合了两张图片，而在mosaic中我们使用四张训练图像按一定比例组合成一张图像，使得模型学会在更小的范围内识别对象，其次还助于显著减少对batch-size的需求，毕竟大多数人的GPU显存有限。

  ![img](https://imgconvert.csdnimg.cn/aHR0cHM6Ly93aWxsaWFtaHlpbi0xMzAxNDA4NjQ2LmNvcy5hcC1zaGFuZ2hhaS5teXFjbG91ZC5jb20vaW1nLzIwMjAwNzEyMTEyOTI3LnBuZw?x-oss-process=image/format,png)

### 自对抗训练（SAT）

Self-Adversarial Training是在一定程度上抵抗对抗攻击的数据增强技术。CNN计算出Loss, 然后通过反向传播改变图片信息，形成图片上没有目标的假象，然后对修改后的图像进行正常的目标检测。需要注意的是在SAT的反向传播的过程中，是不需要改变网络权值的。

使用对抗生成可以改善学习的决策边界中的薄弱环节，提高模型的鲁棒性。因此这种数据增强方式被越来越多的对象检测框架运用。

* ![img](https://imgconvert.csdnimg.cn/aHR0cHM6Ly93aWxsaWFtaHlpbi0xMzAxNDA4NjQ2LmNvcy5hcC1zaGFuZ2hhaS5teXFjbG91ZC5jb20vaW1nLzIwMjAwNzE1MjEzNDEzLnBuZw?x-oss-process=image/format,png)

### 类标签平滑

class label smoothing是以中正规化方法。如果神经网络过度拟合，我们都可以尝试平滑标签，也就是说在训练时标签可能存在错误，而我们可能“过分”相信训练样本的标签，并且在某种程度上没有审视了其他预测的复杂性，因此为了避免过度相信，更合理的做法是对类别标签表示进行编码，以便在一定程度上对不确定进行评估。YOLO V4使用了类平滑，选择模型的正确预测概率为0.9，例如[0,0,0,0.9,0…,0 ]。

![img](https://imgconvert.csdnimg.cn/aHR0cHM6Ly93aWxsaWFtaHlpbi0xMzAxNDA4NjQ2LmNvcy5hcC1zaGFuZ2hhaS5teXFjbG91ZC5jb20vaW1nLzIwMjAwNzE1MjIxMzE4LnBuZw?x-oss-process=image/format,png)

从上图看出，标签平滑为最终的激活产生了更紧密的聚类和更大的类别间的分离，实现更好的泛化。

YOLO V5 似乎没有使用类标签平滑。

### yoloV5数据增强

yolov5都会通过数据加载器传递每一批训练数据，并同时增强数据，数据加载器进行三种数据增强：**缩放，色彩空间调整和马赛克数据增强**。

#### Auto Learning Bounding Box Anchors-自适应锚定框

在yoloV3中，我们采用k均值和遗传学习算法对自定义数据集进行分析，获得适合自定义数据集中对象边界框预测的预设anchor。

![img](https://imgconvert.csdnimg.cn/aHR0cHM6Ly93aWxsaWFtaHlpbi0xMzAxNDA4NjQ2LmNvcy5hcC1zaGFuZ2hhaS5teXFjbG91ZC5jb20vaW1nLzIwMjAwNzEyMTQwNjUwLnBuZw?x-oss-process=image/format,png)

**在yoloV5中anchor是基于训练数据自动学习的**

对于coco数据集来说，yoloV5的配置文件*.yaml中已经预设了640x640图像大小anchor的尺寸。

```
# anchors
anchors:
  - [116,90, 156,198, 373,326]  # P5/32
  - [30,61, 62,45, 59,119]  # P4/16
  - [10,13, 16,30, 33,23]  # P3/8


```

但是对于你的自定义数据集来说，由于目标识别框架往往需要缩放原始图片尺寸，并且数据集中目标对象的大小可能也与COCO数据集不同，因此YOLO V5会重新自动学习锚定框的尺寸。



![img](https://imgconvert.csdnimg.cn/aHR0cHM6Ly93aWxsaWFtaHlpbi0xMzAxNDA4NjQ2LmNvcy5hcC1zaGFuZ2hhaS5teXFjbG91ZC5jb20vaW1nLzIwMjAwNzEyMTQyMDQwLnBuZw?x-oss-process=image/format,png)

如在上图中, YOLO V5在进行学习自动锚定框的尺寸。对于BDD100K数据集，模型中的图片缩放到512后，最佳锚定框为：

![img](https://imgconvert.csdnimg.cn/aHR0cHM6Ly93aWxsaWFtaHlpbi0xMzAxNDA4NjQ2LmNvcy5hcC1zaGFuZ2hhaS5teXFjbG91ZC5jb20vaW1nLzIwMjAwNzEyMTQyMjU0LnBuZw?x-oss-process=image/format,png)

### Backbone-跨阶段局部网络（CSP）

V5和V4中都是用了CSPDarknet作为backbone，从输入图像中提取丰富的信息特征。CSPNet全称是[Cross Stage Partial Networks](https://arxiv.org/pdf/1911.11929.pdf)，也就是跨阶段局部网络。CSPNet解决了其他大型卷积神经网络框架Backbone中网络优化的梯度信息重复问题，将梯度的变化从头到尾地集成到特征图中，因此减少了模型的参数量和FLOPS数值，既保证了推理速度和准确率，又减小了模型尺寸。

CSPNet实际上是基于Densnet的思想，复制基础层的特征映射图，通过dense block 发送副本到下一个阶段，从而将基础层的特征映射图分离出来。这样可以有效缓解梯度消失问题(通过非常深的网络很难去反推丢失信号) ，支持特征传播，鼓励网络重用特征，从而减少网络参数数量。

![img](https://imgconvert.csdnimg.cn/aHR0cHM6Ly93aWxsaWFtaHlpbi0xMzAxNDA4NjQ2LmNvcy5hcC1zaGFuZ2hhaS5teXFjbG91ZC5jb20vaW1nLzIwMjAwNzEyMTUwNTMwLnBuZw?x-oss-process=image/format,png)

CSPNet思想可以和ResNet、ResNeXt和DenseNet结合，目前主要有CSPResNext50 and CSPDarknet53两种改造Backbone网络。

### Neck-PANET

Neck主要用于生成特征金字塔，特征金字塔增强模型对于不同缩放尺度对象的检测，从而能够识别不同大小和尺度的同一个物体。在PANet出来之前，FPN一直对象检测框架聚合层的State of the art,知道panet的出现。在V4的研究中，PANET被认为是最合适yolo的特征融合网络，因此yoloV5和yoloV4都是用了PANET作为neck来聚合特横

PANET基于 Mask R-CNN 和 FPN 框架，同时加强了信息传播。该网络的特征提取器采用了一种新的增强自下向上路径的 FPN 结构，改善了低层特征的传播。第三条通路的每个阶段都将前一阶段的特征映射作为输入，并用3x3卷积层处理它们。输出通过横向连接被添加到自上而下通路的同一阶段特征图中，这些特征图为下一阶段提供信息。同时使用自适应特征池化(Adaptive feature pooling)恢复每个候选区域和所有特征层次之间被破坏的信息路径，聚合每个特征层次上的每个候选区域，避免被任意分配。

![img](https://imgconvert.csdnimg.cn/aHR0cHM6Ly93aWxsaWFtaHlpbi0xMzAxNDA4NjQ2LmNvcy5hcC1zaGFuZ2hhaS5teXFjbG91ZC5jb20vaW1nLzIwMjAwNzEyMTUyNjI1LnBuZw?x-oss-process=image/format,png)

### Head-YOLO通用检测层

模型Head主要用于最终检测部分。它在特征图上应用anchor，并生成带有类概率，对象得分和包围框的最终输出向量。

在 YOLO V5模型中，模型Head与之前的 YOLO V3和 V4版本相同。



![img](https://imgconvert.csdnimg.cn/aHR0cHM6Ly93aWxsaWFtaHlpbi0xMzAxNDA4NjQ2LmNvcy5hcC1zaGFuZ2hhaS5teXFjbG91ZC5jb20vaW1nLzIwMjAwNzE1MjAyMTEzLnBuZw?x-oss-process=image/format,png)

这些不同缩放尺度的Head被用来检测不同大小的物体，每个Head一共(80个类 + 1个概率 + 4坐标) * 3锚定框，一共255个channels。

### 激活函数

激活函数的选择对于深度学习网络是至关重要的。YOLO V5的作者使用了 Leaky ReLU 和 Sigmoid 激活函数。

在 YOLO V5中，中间/隐藏层使用了 Leaky ReLU 激活函数，最后的检测层使用了 Sigmoid 形激活函数。而YOLO V4使用Mish激活函数。

![img](https://imgconvert.csdnimg.cn/aHR0cHM6Ly93aWxsaWFtaHlpbi0xMzAxNDA4NjQ2LmNvcy5hcC1zaGFuZ2hhaS5teXFjbG91ZC5jb20vaW1nLzIwMjAwNzE1MTkzNzM4LnBuZw?x-oss-process=image/format,png)

Mish在39个基准测试中击败了Swish，在40个基准测试中击败了ReLU，一些结果显示基准精度提高了3–5％。但是要注意的是，与ReLU和Swish相比，Mish激活在计算上更加昂贵。

### 优化函数

YOLO V5的作者为我们提供了两个优化函数Adam和SGD，并都预设了与之匹配的训练超参数。默认为SGD。

YOLO V4使用SGD。

YOLO V5的作者建议是，如果需要训练较小的自定义数据集，Adam是更合适的选择，尽管Adam的学习率通常比SGD低。但是如果训练大型数据集，对于YOLOV5来说SGD效果比Adam好。

实际上学术界上对于SGD和Adam哪个更好，一直没有统一的定论，取决于实际项目情况。

## Cost Function-损失函数

YOLO 系列的损失计算是基于 objectness score, class probability score,和 bounding box regression score.

YOLO V5使用 GIOU Loss作为bounding box的损失。

YOLO V5使用二进制交叉熵和 Logits 损失函数计算类概率和目标得分的损失。同时我们也可以使用fl _ gamma参数来激活Focal loss计算损失函数。

YOLO V4使用 CIOU Loss作为bounding box的损失，与其他提到的方法相比，CIOU带来了更快的收敛和更好的性能。

## Summary

总的来说，YOLO V4 在性能上优于YOLO V5，但是在灵活性与速度上弱于YOLO V5。由于YOLO V5仍然在快速更新，因此YOLO V5的最终研究成果如何，还有待分析。我个人觉得对于这些对象检测框架，特征融合层的性能非常重要，目前两者都是使用PANET，但是根据谷歌大脑的研究，BiFPN才是特征融合层的最佳选择。谁能整合这项技术，很有可能取得性能大幅超越。