# PFLD

## 简介

人脸关键点检测也成为人脸对齐，目的是自动定位一组预定义的人脸基准点（比如眼角点/嘴角点）。作为一系列人脸应用的基础，如人脸识别和验证，以及脸部变形和人脸编辑。然而，开发一种使用的人脸关键点检测器仍具有挑战性，因为检测精度，处理速度和模型大小都应该考虑。

在现实世界条件下，获得完美的面孔几乎是不可能的。换句话说，人脸经常是出现在控制不足甚至没有约束的环境中。在不同的照明条件下，它的外表有各种各样的姿势，表情和形状，有时还有部分遮挡。图提供了这样的几个例子。此外，有足够的训练数据用于数据驱动方法也是模型性能的关键。在综合考虑不同条件下，捕捉多个人脸可能是可行的，但这种收集方式会变得不切实际，特别是当需要大规范的数据来训练深度模型时。在这种情况下,我们经常会遇到不平衡的数据分布。本文介绍的这个人脸检测算法PFLD《PFLD: A Practical Facial Landmark Detector》总结了有关人脸关键点检测精度的问题，分为三个挑战（考虑实际使用时，还有一个额外的挑战！）。


![在这里插入图片描述](https://img-blog.csdnimg.cn/2020112222183812.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NTI1MDg0NA==,size_16,color_FFFFFF,t_70#pic_center)

* 局部变化：表情/局部特殊光照/部分遮挡，导致一部分关键点偏离了正常的位置，或者不可见了。
* 全局变化：人脸姿态，成像质量
* 数据不均衡：在人脸数据里面，数据不均衡体现在，大部分是正脸数据，侧脸很少，所以对侧脸，大角度的人脸不太准
* 模型效率：在CNN的解决方案中，模型效率主要是由backbone网络决定。



## 2.网络结构

![在这里插入图片描述](https://img-blog.csdnimg.cn/2020112222262095.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NTI1MDg0NA==,size_16,color_FFFFFF,t_70#pic_center)

**黄色曲线包围的是主网络，用于预测特征点的位置**

**绿色曲线包围的部分为辅助自网络，在训练时预测人脸姿态**（主要包含三个姿态角）

**backbone网络时bottleneck，用MobileNet块代替了传统的卷积运算**。通常这样做，我们的backbone的计算量大大减少，从而加快了速度。此外，可以根据用户需要通过调整Mobilenets的width参数来压缩我们的网络，从而使模型更小，更快。

姿态角的计算方法：

* 预先定义一个标准人脸（在一堆正面人脸上取平均值），在人脸主平面上固定11个关键点作为所有训练人脸的参考
* 使用对应的11个关键点和估计旋转矩阵的参考矩阵
* 由旋转矩阵计算欧拉角



网络结构细节如下：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201122222733879.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NTI1MDg0NA==,size_16,color_FFFFFF,t_70#pic_center)

## 3.损失函数

在深度学习中，数据不平衡时另外一个经常限制准确检测性能的问题。例如，训练集可能包含大量正面，而缺少那些姿势较大的面孔，如果没有额外的技巧，几乎可以肯定的是，由这样的数据集训练的模型不能很好的处理大型姿态情况。而这种情况下，“平均”惩罚每个样本将使其不平等。为了解决这个问题，我们主张对训练样本数量少进行大的惩罚，而不是对样本数据量多多进行惩罚。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201122223108248.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NTI1MDg0NA==,size_16,color_FFFFFF,t_70#pic_center)

![image-20211014201400485](/Users/zhongls/Library/Application Support/typora-user-images/image-20211014201400485.png)


## 4.测试结果

检测精度对比如下面的表所示

![在这里插入图片描述](https://img-blog.csdnimg.cn/2020112222355851.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NTI1MDg0NA==,size_16,color_FFFFFF,t_70#pic_center)

下面来看一下算法处理速度和模型大小，图中C代表i7-6700K CPU,G代表1080 Ti GPU，G*代表Titan X GPU，A代表移动平台Qualcomm ARM 845处理器。

![在这里插入图片描述](https://img-blog.csdnimg.cn/2020112222382758.png#pic_center)

其中PFLD 1X是标准网络，PFLD 0.25X是MobileNet blocks width 参数设为0.25的压缩网络，PFLD 1X+是在WFLW数据集上预训练的网络。

下图是该算法在AFLW数据集上与其他算法的精度比较：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201122224033179.png#pic_center)

消融实验方面作者仅仅分析了损失函数带来的影响，结果如下表所示。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201122224050470.png#pic_center)

## 5.PFLD优化

### 5.1GhostNet

