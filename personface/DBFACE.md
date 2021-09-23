# DBFACE

摘要：基于高斯热力图的目标检测是anchor free中的代表方法，其具有原理简单，易于拓展，后处理简单等优势

## 1. 简介

采用热力图做人脸检测，最开始是[Centernet](https://github.com/xingyizhou/CenterNet)的出现，其在通用**通用目标检测，人体关键点检测，3D目标检测**上都达到了std的效果。后来就出现了[centerface](https://github.com/Star-Clouds/CenterFace)，采用mobilev2做bonenet，做人脸与关键点检测。但，项目只给出了推理代码，没有给出训练代码。后面我借鉴Centernet项目，复现了一个版本：[CenterFace.pytorch](https://github.com/chenjun2hao/CenterFace.pytorch)，但，训练结果没法达到原文的最好效果。且代码是用的Centernet，所以可读性较差。

今年，出现了[DBface](https://github.com/dlunion/DBFace)，和dbface_small，其中dbface_small只有1.4M也能和retinaface_small一样的精度。同时，项目重构的代码，结构清晰，也便于理解和阅读。

## 2. 网络模型结构

### 2.1 神经网络部分

直接上图（PPT画的，😂😂😂）
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200602154922110.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTE2MjIyMDg=,size_16,color_FFFFFF,t_70)

- Block
  就是一个残差模块
- ResSE
  在每个残差模块的shortcut前，增加一个SE模块
- FPN中的UP（借用原作者的经验）
  双线性插值Bilinear+Conv+BN+Activation的上采样模块，优于用反卷积、最近邻插值等，所以比较推荐，但是不同框架对这个的支持不一样，造成使用困难
- SSH
  在输出加了一个SSH模块，增强特征
- 输出
  有三个分支，分别对应**人脸box中心点的高斯热力图，人脸box，人脸关键点landmark**

### 2.2 模型原理部分

* hm loss
  人脸box中心点hm的尺寸为：200*200，hm_target也是200 * 200 ，采用focal loss作为损失函数。hm_target == 1表示正样本，hm_target < 1的表示负样本。所以一个目标用anchor的理念，就只有一个anchor。
  - **引入pos_weight，改善大目标检测**
    CenterNet AnchorFree的特性是所有目标都一个点，正类贡献无论目标尺寸大小都一样，Anchor Base的特性是目标越大正类loss贡献越大，也正因此，大目标在CenterNet上训练效果总是不好，解决大目标不行，本项目提倡增加pos_weights来处理，并且增加大目标的权重，实验证明有良好的提升大目标效果
* box loss——giou loss
  box输出为200 * 200 * 4。在Centernet项目中是预测的w，h，采用的是L1loss。DBface中人脸box采用的是[x1, y1, x2, y2]左上，右下，相对box中心的偏差，并用giou loss。
* landmark loss
  landmark输出为200 * 200 * 10。其也是相对于box中心点的偏差。人脸关键点采用WingLoss









