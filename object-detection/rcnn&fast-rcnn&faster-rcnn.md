### 参考链接

* [RCNN，Fast RCNN，Faster RCNN，MaskRCNN总结](https://blog.csdn.net/qq_38109843/article/details/89320494)
* [目标检测方法-RCNN、fast RCNN、faster RCNN、mask RCNN、SSD、yolo](https://blog.csdn.net/u010801994/article/details/81627757)

* [Faster-RCNN-源码解读](https://zhuanlan.zhihu.com/p/137830097)



# RCNN

选择性搜索selective Search（SS）：

step0：生成区域集R

step1：计算区域集R里每个相邻区域的相似度S={s1,s2...}

step2:找出相似度最高的两个区域，将其合并为新集，添加进R

step3：从S中移除所有与step2中有关的子集

step4：计算新集与所有子集的相似度

step5：跳至step2，直至S为空

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190413095449388.?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM4MTA5ODQz,size_16,color_FFFFFF,t_70)

网络分为4个部分：区域划分，特征提取，区域分类，边框回归

区域划分：使用SS算法画出2K个左右候选框，送入CNN

特征提取：使用imagenet上训练好的模型，进行finetune

区域分类：从头训练一个SVM分类器，对CNN出来的特征向量进行分类

边框回归：使用线性回归，对边框坐标进行精修

# Fast RCNN

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190413101634884.?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM4MTA5ODQz,size_16,color_FFFFFF,t_70)

Fast R-CNN框架与RCNN有两处不同：

1.最后一个卷积层加了一个ROI pooling layer；

2.损失函数使用了multi-task loss（多任务损失）函数，将边框回归直接加到CNN网络中训练。分类Fast RCNN直接用softmax替代RCNN用的SVM进行分类

3.Fast RCNN是端到端（end-to-end）的。



**RCNN和Fast RCNN：**

rcnn：

SS候选框->对每个候选框CNN特征提取->分类+回归

Fast RCNN

一张图片->CNN特征提取->SS->ROI pooling->分类+回归



创新点：

1、fast rcnn实现了大幅度提速，原因整张图像做一次cnn特征提取，通过将候选框映射在conv5输出的feature map上。

2、fast rcnn使用ROI pooling使得尺寸保持一致

3、将分类和回归统一，实现多任务学习

缺点：

候选框的生成仍使用SS方法，无法应用GPU





# Faster RCNN

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190413101929593.?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM4MTA5ODQz,size_16,color_FFFFFF,t_70)

可以看出大体分为4个部分：https://link.zhihu.com/?target=https%3A//mp.weixin.qq.com/s/jhsXSr8xX8YvBK4jIpgX-g

1、conv layers卷积神经网络用于提取特征，得到feature map

2、RPN网络，用于提取ROI

3、ROI  pooling,用于综合ROI和feature map，得到固定大小resize后的feature

4、classifier，用于分类ROI属于哪个类别

## conv layers

在Conv Layers中，对输入的图片进行卷积和池化，用于提取图片特征，最终希望得到的是feature map。在Faster R-CNN中，先将图片Resize到固定尺寸，然后使用了VGG16中的13个卷积层、13个ReLU层、4个maxpooling层。（VGG16中进行了5次下采样，这里舍弃了第四次下采样后的部分，将剩下部分作为Conv Layer提取特征。）

与YOLOv3不同，Faster R-CNN下采样后的分辨率为原始图片分辨率的1/16（YOLOv3是变为原来的1/32）。feature map的分辨率要比YOLOv3的Backbone得到的分辨率要大，这也可以解释为何Faster R-CNN在小目标上的检测效果要优于YOLOv3。

## RPN

简称RPN网络，用于推荐候选区域（Region of Interests），接受的输入为原图片经过Conv Layer后得到的feature map。

![图片](https://mmbiz.qpic.cn/mmbiz_png/SdQCib1UzF3sbfOIDKLCHOegCdHqYg7ibCOGJ5R9Ue9tmLtry8UKvk3l6cDqdjR7pbBxhw65bDIt21k0mqQtW8xg/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

上图参考的实现是：https://github.com/ruotianluo/pytorch-faster-rcnn

RPN网络将feature map作为输入，然后用了3*3卷积将filter减半为512.然后进入两个分支：

一个分支用于计算对应anchor的foreground和background的概率，目标是foreground。

一个分支用于计算对应anchor的Bounding box的偏移量，来获得其目标的定位。

通过RPN网络，我们就得到了每个anchor是否含有目标和在含有目标情况下目标的位置信息。

**对比RPN和YOLOv3:**

都说YOLOv3借鉴了RPN，这里对比一下两者：

**RPN:** 分两个分支，一个分支预测目标框，一个分支预测前景或者背景。将两个工作分开来做的，并且其中前景背景预测分支功能是判断这个anchor是否含有目标，并不会对目标进行分类。另外就是anchor的设置是通过先验得到的。

**YOLOv3**: 将整个问题当做回归问题，直接就可以获取目标类别和坐标。Anchor是通过IoU聚类得到的。

**区别**：Anchor的设置，Ground truth和Anchor的匹配细节不一样。

**联系**：两个都是在最后的feature map（w/16,h/16或者w/32，h/32）上每个点都分配了多个anchor，然后进行匹配。虽然具体实现有较大的差距，但是这个想法有共同点。

### 3. ROI Pooling

这里看一个来自deepsense.ai提供的例子：

RoI Pooling输入是feature map和RoIs：

假设feature map是如下内容：

![图片](https://mmbiz.qpic.cn/mmbiz_png/SdQCib1UzF3sbfOIDKLCHOegCdHqYg7ibCdwbysicz8U8CsXBwzOhYyyKDZmgUN7mPPH6dyicwhQ174X1s5ymCiavRQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

RPN提供的其中一个RoI为：左上角坐标（0,3)，右下角坐标（7,8）

![图片](https://mmbiz.qpic.cn/mmbiz_png/SdQCib1UzF3sbfOIDKLCHOegCdHqYg7ibC5BHjKhhFbKvBSsD9MMZJnj2ybkZclzFIufYm2ePicNS4aVoghmLtqOg/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

然后将RoI对应到feature map上的部分切割为2x2大小的块：

![图片](https://mmbiz.qpic.cn/mmbiz_png/SdQCib1UzF3sbfOIDKLCHOegCdHqYg7ibClQvAViawJIh7Ew9ofXCALuicpkCfia1l3bbKicMhSk5m4uZl1kESuLbXTw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

将每个块做类似maxpooling的操作，得到以下结果：

![图片](https://mmbiz.qpic.cn/mmbiz_png/SdQCib1UzF3sbfOIDKLCHOegCdHqYg7ibCwTgiaer5JIf77JDtxFe7yh2kiaqK17uZxJQyn2WfMwy0CPJiaFsWUJIoQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

以上就是ROI pooling的完整操作，想一想**为何要这样做**？

在RPN阶段，我们得知了当前图片是否有目标，在有目标情况下目标的位置。现在唯一缺少的信息就是这个目标到底属于哪个类别（通过RPN只能得知这个目标属于前景，但并不能得到具体类别）。

如果想要得知这个目标属于哪个类别，最简单的想法就是将得到的框内的图片放入一个CNN进行分类，得到最终类别。这就涉及到最后一个模块：classification

### 4. Classification

ROIPooling后得到的是大小一致的feature，然后分为两个分支，靠下的一个分支去进行分类，上一个分支是用于Bounding box回归。如下图所示（来自知乎）：

![图片](https://mmbiz.qpic.cn/mmbiz_jpg/SdQCib1UzF3sbfOIDKLCHOegCdHqYg7ibCjfFA48wlmPMkDCTHib8YwmiaSmdnOuU2rj1ibN77MCm1j9HD5I7l3QblA/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

分类这个分支很容易理解，用于计算到底属于哪个类别。Bounding box回归的分支用于调整RPN预测得到的Bounding box，让回归的结果更加精确。

 

优化候选框选择算法

具体步骤为：

1、对整张图片进行CNN特征提取，得到feature map

2、feature map输入RPN（region proposal network），快速生成候选区域；（anchor机制）

3、通过交替训练，使得RPN和fast rcnn网络共享参数

4、应用分类和回归

![image](https://img-blog.csdn.net/20180912103750719?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTA4MDE5OTQ=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

# 三者对比

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190413102325747.?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM4MTA5ODQz,size_16,color_FFFFFF,t_70)