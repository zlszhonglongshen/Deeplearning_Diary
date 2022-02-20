# YoloV3-SPP介绍

# 1. YOLO各个版本对比

首先来看下官方给的在COCO数据集上的表现，可以看到SPP版本相对于前面几个版本，mAP有了好几个百分点的提升，在加了SPP模块之后的YOLOv3为何有这么大的提升？



![åä¸ªçæ¬å¨COCOæ°æ®éä¸çè¡¨ç°](https://img-blog.csdnimg.cn/20200215143734935.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5MDU2OTg3,size_16,color_FFFFFF,t_70)

# 2. YOLOv3和YOLOv3-spp网络结构对比

YOLOv3网络结构图：

![YOLOv3ç½ç»ç»æ](https://img-blog.csdnimg.cn/2020021514344999.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5MDU2OTg3,size_16,color_FFFFFF,t_70)

YOLOv3-spp网络结构图：

![YOLOv3-sppç½ç»ç»æ](https://img-blog.csdnimg.cn/20200215143519584.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5MDU2OTg3,size_16,color_FFFFFF,t_70)

SPP模块结构如下图：

![å¨è¿éæå¥å¾çæè¿°](https://img-blog.csdnimg.cn/20200215144249398.png)

从上述两幅网络结构图中我们可以看出，在相对于普通版本的yoloV3，SPP版本在第五，六层卷积之间增加了一个SPP模块，这个模块主要是由不同的池化操作组成，具体的实现在yoloV3-SPP的cfg文件中：

```
### SPP ###
[maxpool]
stride=1
size=5

[route]
layers=-2

[maxpool]
stride=1
size=9

[route]
layers=-4

[maxpool]
stride=1
size=13

[route]
layers=-1,-3,-5,-6

```

# 3. 如何理解yolov3中的spp模块？

在一般的CNN网络结构中，最后的分类层通常是由全连接层组成，而全连接有个特点，就是他特征数是固定的，这就导致了图片在输入网络的时候，大小必须是固定的，但是在实际情况中，图片大小是多种多样的，如果不能满足网络的输入，图片将无法在网络中进行前向运算，所以为了得到固定尺寸的图片，必须对图片进行裁剪或者变形拉伸等，这样就很可能会导致图像失真，从而影响最终的精度，而我们希望网络能够保持原图大小的输入，得到最大的精度。

![å¨è¿éæå¥å¾çæè¿°](https://img-blog.csdnimg.cn/20200215153435379.png)

SPP全称为Spatial Pyramid Pooling（空间金字塔池化结构），它是由微软研究院的何凯明大神提出，主要是为了解决两个问题：

1、有效避免了对图像区域裁剪，缩放操作导致的图像失真等问题。

2、解决了卷积神经网络对图像重复特征提取的问题，大大提高了产生候选框的速度，且节省了计算成本

但是在yoloV3中，并不是解决这两个问题的，如果对于如何解决上述问题感兴趣的同学，可以去参考[这篇文章](https://blog.csdn.net/yzf0011/article/details/75212513)在YOLOv3-SPP中，SPP module由四个并行的分支构成，分别是kernel size为 5×5, 9×9, 13×13的最大池化和一个跳跃连接。如下图所示，作者检测头前面的第5和第6卷积层之间集成SPP模块来获得YOLOv3-SPP，在Feature Map经过SPP module池化后的特征图重新cat起来传到下一层侦测网络中
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200215152753520.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5MDU2OTg3,size_16,color_FFFFFF,t_70)

所以说，yoloV3-spp版本实际上只是增加了SPP模块，该模块借鉴了空间金字塔的思想，通过SPP模块实现了局部特征和全局特征，这也是为什么SPP模块中最大的池化核大小要尽可能的借鉴或者等于需要池化的特征图的大小，特征图经过局部特征与全局特征相融合后，丰富了特征图的表达能力，有利于待检测图像中目标大小差异较大的情况，尤其是对于yoloV3中这种复杂的多目标检测，所以对检测的精度上有了很大的提升。

# 参考文档：

https://blog.csdn.net/yzf0011/article/details/75212513
https://blog.csdn.net/qq_33270279/article/details/103898245
https://zhuanlan.zhihu.com/p/78942216
