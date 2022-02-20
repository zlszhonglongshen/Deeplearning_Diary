# mAP详解和计算方式

## 1.理解mAP前要知道的一些基础概念

### IOU

IOU的全称为交并比，在目标检测中即为计算预测边界框与真实边界框的重叠度，重叠度越高，说明越接近真是框。IOU计算的是“预测的边框”和“真实的边框”的交集和并集的比值，即公式等于：

**IOU=想交的面积/相并的面积**

![IOU计算公式](https://img-blog.csdnimg.cn/20200216213445948.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5MDU2OTg3,size_16,color_FFFFFF,t_70)

### 混淆矩阵

混淆矩阵是机器学习中总结分类模型预测结果的情形分析表，以矩阵形式将数据集中的记录按照真实的类别与分类模型预测的类别判断两个标准进行汇总。其中矩阵的行表表示真实值，矩阵的列表示预测值。矩阵表现形式，如下：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200216210233567.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5MDU2OTg3,size_16,color_FFFFFF,t_70)

为了更好的理解，我们把矩阵中英文翻译成字面信息，True（对）、False（错）、Positive（正例）、Negative（负例），混淆矩阵重新整理一下：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200216210854502.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5MDU2OTg3,size_16,color_FFFFFF,t_70)

现在我们来分析一下这个混淆矩阵，在mAP计算中，混淆矩阵的概念是非常重要的。

2.1 真正例 TP（True Positives）：

预测结果是True（对），真实值为Positives（正例），模型认为它也是Positives（正例）；在一般情况下，会认为IOU>0.5时为真正例，在一些数据集上的指标也是为0.5；

2.2 假正例 FP（False Positives）

预测结果是False（错），真实值为Negative（负例），模型却认为它是Positives（正例）；同理，一般情况下会认为 IOU<0.5 时为假正例；

2.3 假反例 FN（ False Negatives）

预测结果是False（错），真实值为Positives（正例），模型预测它为Negative（负例）；

2.4 真反例 TN（True Negatives）

预测结果为True（对），真实值为Negative（负例），模型预测它为Negative（负例）；一般用不到这个

理解了混淆矩阵里的元素概念后，我们就可以继续了解以下概念


**2.5 准确率（Accuracy）**

即表示分类模型所判断的**所有结果中**，预测正确的结果占比，公式为：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200216220812397.png)
一般不会作为模型的评估标准；

**2.6 精准率或者查准率（precision）**

即表示在**所有预测值为正例中**，有多少正例被预测出来，计算公式为：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200216215503946.png)
相对于准确率来说，对模型的好坏评估结果会更好，注意和上面的准确率区分；

**2.7 召回率（Recall）**

即表示**所有真实值为正例中**有多少被预测出来，可以理解为有多少正确的目标被召回（找出），计算公式为：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200216215830478.png)
在一些特定场景中，会将其作为模型评估标准；

#### 2.8平均精度AP（Average Precision)

PR曲线（红线）以下与横轴、纵轴之间的面积。PR曲线是由Precision（精准率或者查准率）与Recall（召回率或者查全率）构成的曲线，横轴为Recall，纵轴为Precision。

![å¨è¿éæå¥å¾çæè¿°](https://img-blog.csdnimg.cn/20200216221837732.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5MDU2OTg3,size_16,color_FFFFFF,t_70)

#### 2.9 **mAP（mean Average Precision）**

AP衡量的是对一个类检测好坏，mAP就是对多个类的检测好坏。在多类多目标检测中，计算出每个类别的AP后，再除于类别总数，即所有类别AP的平均值，比如有两类，类A的AP值是0.5，类B的AP值是0.2，那么mAP=（0.5+0.2）/2=0.35。

## mAP计算过程

要计算mAP，首先要计算每一类的AP，比我我们有一个项目，有一个类别检测的是人脸，模型训练完后，我们有一组测试图片，每张图片上都有已经标记好的人脸 label ，这样我们就有了人脸的真实边界框（Ground Truth），在每张图片输入模型后会得到一系列人脸类别的预测边界框，每一个框上面都有一个置信度（暂时不考虑其他类别的框）。

将每张测试图片进行检测后，会得到一系列预测边界框集合，然后将这个预测边界框集合按照置信度降序排序。对于某一张测试图片，我们先计算该图片上人脸类别的预测边界框和真实的边界框的重叠度（IOU），当重叠度（IOU)大于设定的阈值（一般为0.5，可以自己设置）则将该边界框记作真正例（TP），否则记为假正例（FP）。对于测试集中的每一张图片均进行上述的操作（注意：在计算某一张图片的预测框是否为TP时，会从预测框集合中选取出该图片的预测框和真实框做对比）由此可以判定预测边界框集合中的所有预测框属于TP或者是FP。

比如下面三张测试图片，检测的是人脸，绿色框表示真实边界框，红色框表示预测边界框，旁边的红色数字为置信度。


比如下面三张测试图片，检测的是人脸，绿色框表示真实边界框，红色框表示预测边界框，旁边的红色数字为置信度。

![å¨è¿éæå¥å¾çæè¿°](https://img-blog.csdnimg.cn/20200216225527556.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5MDU2OTg3,size_16,color_FFFFFF,t_70)

![å¨è¿éæå¥å¾çæè¿°](https://img-blog.csdnimg.cn/20200216225559303.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5MDU2OTg3,size_16,color_FFFFFF,t_70)

![å¨è¿éæå¥å¾çæè¿°](https://img-blog.csdnimg.cn/20200216225943371.jpeg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5MDU2OTg3,size_16,color_FFFFFF,t_70)

可以得出三个真实边界框（GT1、GT2、GT3），三个预测框边界框（BBox1，BBox2， BBox3）。

1.首先按照置信度进行降序排序；
2.对于每张图片中的预测框计算IOU，可以很清楚的看出：BBox1为TP，BBox2为FP，BBox3为TP；
3.之后计算不同召回率（Recall）下的精准率（Precision）值。对于第一个预测边界框BBox1，其Precision = TP / (TP+NP)=1/（1+0）=1，Recall = TP / GT(总真实框数）= 1/3，同理，排序后的前两个预测边界框BBox1、BBox3，计算Precision = 1+1 / （1+1）= 1 ，Recall = 2/3，接下来三个预测边界框BBox1、BBox2、BBox3，Precision = 1+1 / （1+1+1）= 2/3 ，Recall = 2/3，这样我们就有了一组Precision、Recall值[（1，1/3），（1，2/3），（2/3，2/3）]
4.绘制PR曲线如下图，然后每个“峰值点”往左画一条线段直到与上一个峰值点的垂直线相交。这样画出来的黄色线段与坐标轴围起来的面积就是AP值。这里
AP = （1/3 - 0）x 1 + （2/3 - 1/3）x 1 + （1 - 2/3）x 0 = 0.667

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200217000539156.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5MDU2OTg3,size_16,color_FFFFFF,t_70)

按照如上方法计算其他所有类的AP，最后取平均值即为mAP

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200217000647970.png)

其中C表示总类别数目，APi表示第i类的AP值。

## 三、例子（AP计算）

再通过个例子来更好的理解mAP计算过程，该部分摘自这篇博客，原文链接：[目标检测中的评价指标mAP理解及计算](https://blog.csdn.net/NooahH/article/details/90140912)

比如说我们的测试集中类A的GT（真实框）有7个，经过目标检测模型预测到了10个边界框，经过上次排序及判断操作，有如下结果：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200217145138417.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5MDU2OTg3,size_16,color_FFFFFF,t_70)

按照confidence（置信度）降序排序。从上表TP可以看出我们预测正确5个（TP=5），从FP看出预测错误5个（FP=5）。除了表中已预测到的5个GT，还有2个GT并未被预测出来（FN=2）。
接下来计算AP，计算前*个BBox得到的precision和recall：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200217145305862.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5MDU2OTg3,size_16,color_FFFFFF,t_70)

在计算precision和Recall时Rank指的是前个预测边界框的TP和FP之和。

于是我们得到了一个recall阈值列表[0,0.14,0.29,0.43,0.57,0.71,1]，为recall阈值列表中的各值生成对应的precision列表，选择recall>=阈值元素所对应的precision的最大值。为此可得precision列表为[1,1,1,0.5,0.5,0.5,0]。

在这里举个计算的例子吧，比如找recall阈值列表中0.57所对应的precision，当recall>=0.57时，由上表可得precision为max{0.44,0.5}=0.5，其他recall对应precision的选取同理。

有了这两个列表就可以计算类A的AP了：
AP=(0.14−0)∗1+(0.29−0.14)∗1+(0.43−0.29)∗0.5+(0.57−0.43)∗0.5+(0.71−0.57)∗0.5+(1−0.71)∗0=0.5。

同样可以通过绘制PR曲线计算线下面积，如下图所示：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200217145357599.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5MDU2OTg3,size_16,color_FFFFFF,t_70)

AP值即浅蓝色图形的面积，蓝色折线为recall、precision点，同理求出其他类的AP，即可算出mAP值。