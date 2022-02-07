## 参考链接

* [【从零开始学Mask RCNN】一，原理回顾&&项目文档翻译](https://zhuanlan.zhihu.com/p/150104383)
* [Mask RCNN原理讲解](https://blog.csdn.net/qq_37392244/article/details/88844681)
* [令人拍案称奇的Mask RCNN](https://zhuanlan.zhihu.com/p/37998710)

### 原理回顾

### 1.前言
mark-rcnn是一C实例分割框架，通过增加不同的分支可以完成目标分类，目标检测，语义分割，人体姿态估计等多种任务。对于实例分割来讲，就是在Faster-RCNN的基础上（分类+回归）增加了一个分支用于语义分割，其抽象机构如figure1所示：

![img](https://pic2.zhimg.com/80/v2-633bc797dcc90a23ee2d46c6365ddd21_720w.jpg)
稍微描述下这个结构：

* 输入预处理后的原始图片
* 将输入图片送入到特征提取网络得到特征图
* 然后对特征图的每一个像素位置设定固定个数的ROI（也可以叫anchor），然后将ROI区域送入RPN网络进行二分类（前背景和后背景）以及坐标回归，以获得精炼后的ROI区域。
* 对上个步骤中获得的ROI区域执行论文提出的ROIAlign操作，即先将原图和feature map的pixel对应起来，然后将feature map和固定的feature对应起来。
* 最后对这些ROI区域进行多类别分类，候选框回归以及引入FCN生产mask，完成分割任务。
* ![img](https://pic3.zhimg.com/80/v2-bc1489f2bfa3f4d9be094ad8476103d2_720w.jpg)

### 2.方法

#### 2.1原始ROI pooling的问题

在faster-RCNN中ROI pooling的过程如下图所示：

![img](https://pic2.zhimg.com/80/v2-29cf5852c410c5d38532e9ad0de2879d_720w.jpg)

输入图片的大小为![[公式]](https://www.zhihu.com/equation?tex=800%5Ctimes+800)，其中狗这个目标框的大小为![[公式]](https://www.zhihu.com/equation?tex=665%5Ctimes+665)，经过VGG16网络之后获得的特征图尺寸为![[公式]](https://www.zhihu.com/equation?tex=800%2F32+%5Ctimes+800%2F32%3D25%5Ctimes+25)，其中![[公式]](https://www.zhihu.com/equation?tex=32)代表VGG16中的![[公式]](https://www.zhihu.com/equation?tex=5)次下采样（步长为2）操作。同样，对于狗这个目标，我们将其对应到特征图上得到的结果是![[公式]](https://www.zhihu.com/equation?tex=665%2F32+%5Ctimes+665%2F32%3D20.78%5Ctimes+20.78%3D20%5Ctimes+20)，因为坐标要保留整数所以这里引入了第一个量化误差即舍弃了目标框在特征图上对应长宽的浮点数部分。

然后我们需要将这个![[公式]](https://www.zhihu.com/equation?tex=20%5Ctimes+20)的ROI区域映射为![[公式]](https://www.zhihu.com/equation?tex=7%5Ctimes+7)的ROI特征图，根据ROI Pooling的计算方式，其结果就是![[公式]](https://www.zhihu.com/equation?tex=20%2F7+%5Ctimes+20%2F7%3D2.86%5Ctimes+2.86)，同样执行取整操作操作后ROI特征区域的尺寸为![[公式]](https://www.zhihu.com/equation?tex=2%5Ctimes+2)，这里引入了第二次量化误差。

从上面的分析可以看出，这两次量化误差会导致原始图像中的像素和特征图中的像素进行对应时出现偏差，例如上面将![[公式]](https://www.zhihu.com/equation?tex=2.86)量化为![[公式]](https://www.zhihu.com/equation?tex=2)的时候就引入了![[公式]](https://www.zhihu.com/equation?tex=0.86)的偏差，这个偏差映射回原图就是![[公式]](https://www.zhihu.com/equation?tex=0.86%5Ctimes+32%3D27.52)，可以看到这个像素偏差是很大的。

#### 2.2ROIAlign

为了缓解ROI Pooling量化误差过大的缺点，文本提出ROIAlign，ROIAlign没有使用量化操作，而是使用了双线性插值。它充分的利用原图中的虚拟像素值如27.52四周的四个真实存在的像素值来共同决定目标图中的一个像素值。即它可以将和27.52类似的非整数坐标值像素对应的输出像素估计出来。这一过程如下图：

![img](https://pic4.zhimg.com/80/v2-20ecbdfc6c0ccca67f24213bde72f743_720w.jpg)

其中feat.map就是VGG16或者其他Backbone网络获得的特征图，黑色实线表示的是ROI划分方式，最后输出的特征图大小为![[公式]](https://www.zhihu.com/equation?tex=2%5Ctimes+2)，然后就使用双线性插值的方式来估计这些蓝色点的像素值，最后得到输出，然后再在橘红色的区域中执行Pooling操作最后得到![[公式]](https://www.zhihu.com/equation?tex=2%5Ctimes+2)的输出特征图。可以看到，这个过程相比于ROI Pooling没有引入任何量化操作，即原图中的像素和特征图中的像素是完全对齐的，没有偏差，这不仅会提高检测的精度，同时也会有利于实例分割。

#### 2.3网络结构

为了证明次网络的通用性，论文构造了多种不同结构的Mask R-CNN，具体为使用**不同的**Backbone网络以及**是否**将用于边框识别和Mask预测的**上层**网络分别应用于每个ROI。对于Backbone网络，Mask R-CNN基本使用了之前提出的架构，同时添加了一个全卷积的Mask(掩膜)预测分支。Figure3展示了两种典型的Mask R-CNN网络结构，左边的是采用![[公式]](https://www.zhihu.com/equation?tex=ResNet)或者![[公式]](https://www.zhihu.com/equation?tex=ResNeXt)做网络的backbone提取特征，右边的网络采用FPN网络做Backbone提取特征，最终作者发现**使用ResNet-FPN作为特征提取的backbone具有更高的精度和更快的运行速度**，所以实际工作时大多采用右图的完全并行的mask/分类回归。

![img](https://pic2.zhimg.com/80/v2-39be7259f4dd4eef8a778cd621fe3879_720w.jpg)											Mask RCNN的两种经典结构

#### 2.4损失函数

Mask分支针对每个ROI区域产生一个![[公式]](https://www.zhihu.com/equation?tex=K%5Ctimes+m%5Ctimes+m)的输出特征图，即![[公式]](https://www.zhihu.com/equation?tex=K)个![[公式]](https://www.zhihu.com/equation?tex=m%5Ctimes+m)的二值掩膜图像，其中![[公式]](https://www.zhihu.com/equation?tex=K)代表目标种类数。Mask-RCNN在Faster-RCNN的基础上多了一个ROIAligin和Mask预测分支，因此Mask R-CNN的损失也是多任务损失，可以表示为如下公式： ![[公式]](https://www.zhihu.com/equation?tex=L%3DL_%7Bcls%7D%2BL_%7Bbox%7D%2BL_%7Bmask%7D) 其中![[公式]](https://www.zhihu.com/equation?tex=L_%7Bcls%7D)表示预测框的分类损失，![[公式]](https://www.zhihu.com/equation?tex=L_%7Bbox%7D)表示预测框的回归损失，![[公式]](https://www.zhihu.com/equation?tex=L_%7Bmask%7D)表示Mask部分的损失。 对于预测的二值掩膜输出，论文对每一个像素点应用`sigmoid`函数，整体损失定义为平均二值交叉损失熵。引入预测K个输出的机制，允许每个类都生成独立的掩膜，避免类间竞争。这样做解耦了掩膜和种类预测。不像FCN的做法，在每个像素点上应用`softmax`函数，整体采用的多任务交叉熵，这样会导致类间竞争，最终导致分割效果差。

下图更清晰的展示了Mask-RCNN的Mask预测部分的损失计算，来自知乎用户`vision`：

![img](https://pic4.zhimg.com/80/v2-10ddcfcedc29f2c7fb71cfa154e15f53_720w.jpg)Mask-RCNN的Mask预测部分的损失计算



# Mark RCNN

1：简介

mask RCNN可以看做是一个通用实例分割架构。Mask RCNN以faster-rcnn原型，增加了一个分支用于分割任务，对于Faster-RCNN的每个proposal box都要使用FCN进行语义分割，分割任务与定位、分类任务是同时进行的。引入了ROi Align代替Faster RCNN中的ROI Pooling。因为Roi Pooling并不是按照像素一一对齐的，也许这对bbox的影响不是很多大，但对于mask的精度却有很大影响。引入语义分割分支，实现了mask和class预测的关系的解耦，mask分支只做语义分割，类型预测的任务交给另一个分支。这与原本的FCN网络是不同的。原始的FCN在预测时还用同时预测mask所属的种类。



![在这里插入图片描述](https://img-blog.csdnimg.cn/20190416162658291.?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM4MTA5ODQz,size_16,color_FFFFFF,t_70)

