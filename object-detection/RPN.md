# RPN区域推荐网络





-- https://mp.weixin.qq.com/s/sN6WmeWD_WTZVLhMP4uRbA



###  边框的位置到底用什么表示？

 目标检测之所以难，是因为每一个物体的区域大小是不一样的。每一个区域有着不同的大小size（scale）和不同的长宽比（aspect ratios）。

 现在假设，已经知道图片中有两个objects，首先想到的是，训练一个网络，输出8个值：两对元组（xmin,ymin,xmax,ymax)，（xmin,ymin,xmax,ymax)分别定义了每个object的边界框，这种方法存在一些基本问题。例如，

（1）当图片的尺寸和长宽不一致时，良好训练模型来预测，会非常复杂；

（2）另外一个问题是无效预测，预测xmin和xman时，需要保障xmin<xmax；

 事实上，有一种更加简单的方法来预测objects的边界框，即，学习相对于参考boxes的偏移量，假设参考box的位置由以下确定：（xcenter，ycenter,width,height).，则需要预测量为：（ Δxcenter,Δycenter,Δwidth,Δheight），它们的取值一般都是很小的值，以调整参考 box 更好的拟合所需要的。

### RPN网络的基本结构





![img](https://mmbiz.qpic.cn/mmbiz_png/KeMbZlA1x1b7icTj8hI2vdkgnW79YvNbpaRrqFxr4ibqQHicfvu663V2t8WiaY6LWMeOeFcpF3iauPsxUbXFcG7ibC1g/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)





### RPN的核心概念-anchor





anchor，中文意思是锚点的意思，那到底什么是锚点？有很多不同的解释，其实不能说哪种解释完全标准，主要是站在不同的角度进行解释的。都有道理，本文会将各种解释综合起来，尽量说的清晰。



#### 其他概念



要理解什么是anchor，就要知道我们为什么要高出一个RPN这种东西。



（1）RCNN是在原始图像中使用select search方法选择大约2000个候选框，然后对每个候选框进行卷积运算。

（2）sppnet和fast-rcnn是让整个图片经过cnn，然后在得到的特征图上使用select search选择2000个左右候选框，其实我们真正需要的还是在原始图像上的候选区域，那为什么在卷积之后的特征图上也可以这么做呢？这是因为卷积之后的特征图和原始图像之间存在的映射关系，在特征图上的候选区可以映射到原始图像上。





总而言之：我们要使用RPN的目的也就是要产生（原始图像上的）候选框。而且这里有一个重要的信息，就是卷积之后的feature map和原始图像之间是一个映射关系的，如果我能够在feature map上面找到相应候选框，也就代表了原始图像上找到了候选框。





### RPN的输入和输出



输入：RPN是接在feature map之后的，因此它的输入是feature map

输出：我希望得到的是候选区域，因为输出的是候选区域，这样说没错，但是在网络中其实流动的都是数据啊，这样一个框框是怎么表示？当然也是通过数据的形式来表示了。还有一点就是这个框框里面有目标还是没有目标，这也是通过数据来表示的。





### 到底什么是anchor

有很多文献中说到，anchor是大小和尺寸固定的候选框，个人感觉这种说法不是很准确。只是一个表现而已。我们先看下RPN网络的一个第一步运算，RPN的第一步运算实际上就是一个3x3x256的卷积运算，我们称3x3为一个滑动窗口，假设RPN的输入是13x13x256的特征图，然后使用3x3x256的卷积核进行卷积运算，最后依然会得到一个axax256的特征图，这里的a与卷积的步幅有关。

在原始论文中，我们选定了3中不同scale，3中不同长宽比的矩形框作为基本候选框。

三种scale/size是{128,256,512}

三种比例{1:1,1:2,2:1}

故而一共是3x3=9种，有很多文献说这就是9个anchors，之所以我觉得不准确是因为下面的两个方面

（1）anchor顾名思义为锚点，这这是一个矩形框，与锚点本身的含义是不符合的；

（2）很明显，这9个基本候选框的长宽远远大于特征图的长宽，所以这9个指的应该是原始图像，结合论文中要对原始图像进行缩放到600*1000左右的大小，更加确定了这一点，有的人说锚点是特征图上的某一个点或者是候选框，既然这9个根本就不是特征图上的候选框，那自然不存在锚点之说了。



### **anchor锚点的本质**

锚点的真实含义：应该是特征图的某一个像素与对应在原始图像的某一个像素，即它本质上指的是特征图上当前滑窗的中心在原像素空间的应设点成为anchor，即anchor是在原始图像上的，然后以这个锚点为中心，配上9个基本候选框，这就正确了，所以在原始图像大致如下：





![img](https://mmbiz.qpic.cn/mmbiz_png/KeMbZlA1x1b7icTj8hI2vdkgnW79YvNbpedPwkeAFKCn7T5IFyibvVAD5KREqEEvWSpCR73nib0kEoy2cibc2csBzw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)





其中一个黑点的点是anchor锚点啊，自己画出的以这些锚点为中心的9个基本候选框。





### 为什么这样子的设计可行？





依然以上面的例子进行说明，假定输出特征图为13x13x256，然后在该特征图进行3x3x256的卷积，默认进行了边界填充。



那么每一个特征图上一共有13x13=169个像素点。由于采用了边界填充，所以在进行3x3卷积的时候，每一个像素点都可以做一次3x3卷积核的中心点，那么整个卷积下来相当于有169个卷积中心，这169个卷积中心在原始图像上会有169个对应的锚点，然后每个锚点有9个默认大小的基本候选框，这就相当远原始图像中一共有169x9=1521个候选框，这1521个候选框有9种不同的尺度，中心又到处都分布，所以足以覆盖了整个原始图像上所有的区域，甚至还有大量的重复区域。





![img](https://mmbiz.qpic.cn/mmbiz_png/KeMbZlA1x1b7icTj8hI2vdkgnW79YvNbp6Q6bs1d6yD6LSibArsqtEdKV5ia5GiaYibmKfM6vw5TkmV3XDelZwje1sw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)





**总结归纳：**





   当前滑窗的中心在原像素空间的映射点称为anchor，以此anchor为中心，生成k(paper中default k=9, 3 scales and 3 aspect ratios)个proposals。虽然 anchors 是基于卷积特征图定义的，但最终的 anchos 是相对于原始图片的.





**RPN的本质是 “ 基于滑窗的无类别obejct检测器 ” 。**







## 生成anchor的作用和目的





​    要知道，训练RPN网络是有监督训练，需要有数据、还要有相应的类标签，输入小网络的是512个通道的3*3滑窗，类标签没有给定，没有类标签就无法计算Loss损失函数，无法训练网络。以3*3滑窗中心对应原图的位置作为中心点，在原图生成9个不同尺度长宽比的anchor，然后每个anchor都会被分配到相应的类标签，有正样本(1)、负样本(0)，也有不参与训练的框（not used），对正样本计算，就是回归的类标签，负样本不计算回归loss。0,1是二分类的标签。所以在原图生成anchor的目的之一是得到类标签。这里只得到了分类的标签（0，1），还有正样本的回归标签需要确定，该正样本的回归标签是其对应的ground truth计算出来的。负样本不计算回归损失没有回归标签。 





**RPN的训练过程**



上面只讨论了RPN的第一步运算——实际上就是卷积运算，接下来考虑后面的运算步骤，如下：





![img](https://mmbiz.qpic.cn/mmbiz_png/KeMbZlA1x1b7icTj8hI2vdkgnW79YvNbpUibzzkAJ3KHVhFYHW5ia0daPfyXPibSHCV6iaG86tsSTqskcJ5TjxhuZjQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)





上面的用黑色圈出来的部分是第一步运算，用红色圈圈圈出来的是第二步运算，我们将第二步运算单独拿出来看，如下图所示：





![img](https://mmbiz.qpic.cn/mmbiz_jpg/KeMbZlA1x1b7icTj8hI2vdkgnW79YvNbpy7HDR4UdiaqdtDvrJiaxeaWkH8SDBwQUbR8bV9l7Siccy6OvYA2PMqmhw/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)





注意：这里图片的和上面的稍微有所区别，但是没关系，只要能理解清楚意思就好。





   RPN 是全卷积(full conv) 网络，其采用基础网络输出的卷积特征图作为输入. 首先，采用 512 channel，3×3 kernel 的卷积层（上面的例子采用的是256个channel，但是不影响理解），然后是两个并行的 1×1 kernel 的卷积层，该卷积层的 channels 数量取决每个点（每个anchor）所对应的的 标准候选框K 的数量，在这两个并行的1x1卷积中，左侧的是进行分类的，这里的分类只是分有和无两类，即候选框中有目标还是没有目标，至于目标到底是什么先不管，由于每一个anchor对应k个候选框，每一个候选框有两个取值（即有和无，用概率大小表示）所以每一个anchor对应的输出应该为一个2K维度的向量，故而左侧的分类卷积使用2K个channel；





​    同样的道理，右侧是获取边框位置信息的卷积网络，由于每一个anchor对应k个候选框，每一个候选框有4个位置取值（x,y,w,h）所以每一个anchor对应的输出应该为一个4K维度的向量，故而右侧的卷积使用4K个channel；这里的理解是很重要的。





那究竟RPN网络是如何进行训练的呢？





  RPN训练中对于正样本文章中给出两种定义。第一，与ground truth box有最大的IoU的anchors作为正样本；第二，与ground truth box的IoU大于0.7的作为正样本。文中采取的是第一种方式。文中定义的负样本为与ground truth box的IoU小于0.3的样本。 





训练RPN的loss函数定义如下： 





![img](https://mmbiz.qpic.cn/mmbiz_png/KeMbZlA1x1b7icTj8hI2vdkgnW79YvNbpX0TdibWKYD8rqtWrM5rHa7W1gtDAMwdKwsa8TRxnAc6qbImJNBZHK8g/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)





   其中，i表示mini-batch中第i个anchor，pi表示第i个anchor是前景的概率，当第i个anchor是前景时pi为1反之为0，ti表示预测的bounding box的坐标，ti∗为ground truth的坐标。 





   看过Fast R-CNN文章详细解读文章的会发现，这部分的loss函数和Fast R-CNN一样，除了正负样本的定义不一样，其他表示时一样的。一个是交叉熵损失，一个是smooth_L1损失函数。





**RPN是如何产生ROI的？**





  RPN在自身训练的同时，还会提供RoIs（region of interests）给Fast RCNN（RoIHead）作为训练样本。RPN生成RoIs的过程(ProposalCreator)如下：





（1）对于每张图片，利用它的feature map， 计算 (H/16)× (W/16)×9（大概20000）个anchor属于前景的概率，以及对应的位置参数。（这里的W、H表示原始图像的宽和高，前面已经有说过了）





（2）选取概率较大的12000个anchor，利用回归的位置参数，修正这12000个anchor的位置，得到RoIs，利用非极大值（(Non-maximum suppression, NMS）抑制，选出概率最大的2000个RoIs





**注意：**在inference的时候，为了提高处理速度，12000和2000分别变为6000和300.





**注意：**这部分的操作不需要进行反向传播，因此可以利用numpy/tensor实现。





RPN的输出：RoIs（形如2000×4或者300×4的tensor）





 **RPN网络与Fast R-CNN网络的权值共享** 





   RPN最终目的是得到候选区域，但在目标检测的最终目的是为了得到最终的物体的位置和相应的概率，这部分功能由Fast R-CNN做的。因为RPN和Fast R-CNN都会要求利用CNN网络提取特征，所以文章的做法是使RPN和Fast R-CNN共享同一个CNN部分。 





   Faster R-CNN的训练方法主要分为两个，目的都是使得RPN和Fast R-CNN共享CNN部分，如下图所示 ：





![img](https://mmbiz.qpic.cn/mmbiz_png/KeMbZlA1x1b7icTj8hI2vdkgnW79YvNbpIgFyvayK1ZKv8iblxnj04icQt8jyjeTtOcRrbfcFa4yV6ibgG268h2icyA/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)





​    一个是迭代的，先训练RPN，然后使用得到的候选区域训练Fast R-CNN，之后再使用得到的Fast R-CNN中的CNN去初始化RPN的CNN再次训练RPN(这里不更新CNN，仅更新RPN特有的层)，最后再次训练Fast R-CNN(这里不更新CNN，仅更新Fast R-CNN特有的层)。 





​    还有一个更为简单的方法，就是end-to-end的训练方法，将RPN和Fast R-CNN结合起来一起训练，tf版本的代码有这种方式的实现。







# RPN

-- https://segmentfault.com/a/1190000021742853?utm_source=tag-newest



![img](https://img2018.cnblogs.com/blog/1414369/201903/1414369-20190323160415685-582962582.png)





## RPN的运作机制



![img](https://img2018.cnblogs.com/blog/1414369/201903/1414369-20190323160613631-1810905247.png)



图2展示了RPN的整个过程，一个特征图经过sliding window处理，得到256维特征，然后通过两次全连接得到结果2k个分数和4k个坐标；相信大家一定有很多不懂的地方；我把相关的问题一一列举：



\1. RPN的input 特征图指的是哪个特征图？



\2. 为什么是用sliding window？文中不是说用CNN么？



\3. 256维特征向量如何获得的？



\4. 2k和4k中的k指的是什么？



\5. 图右侧不同形状的矩形和Anchors又是如何得到的？



   



　首先回答第一个问题，RPN的输入特征图就是图1中Faster RCNN的公共Feature Map，也称共享Feature Map，主要用以RPN和RoI Pooling共享；



　　对于第二个问题，我们可以把3x3的sliding window看作是对特征图做了一次3x3的卷积操作，最后得到了一个channel数目是256的特征图，尺寸和公共特征图相同，我们假设是256 x （H x W）；



 对于第三个问题，我们可以近似的把这个特征图看作有H x W个向量，每个向量是256维，那么图中的256维指的就是其中一个向量，然后我们要对每个特征向量做两次全连接操作，一个得到2个分数，一个得到4个坐标，由于我们要对每个向量做同样的全连接操作，等同于对整个特征图做两次1 x 1的卷积，得到一个2 x H x W和一个4 x H x W大小的特征图，换句话说，有H x W个结果，每个结果包含2个分数和4个坐标；



![img](https://img2018.cnblogs.com/blog/1414369/201903/1414369-20190323160837097-1235542204.png)



这里我们需要解释一下为何是2个分数，因为RPN是提候选框，还不用判断类别，所以只要求区分是不是物体就行，那么就有两个分数，前景（物体）的分数，和背景的分数； 

　　我们还需要注意：4个坐标是指针对原图坐标的偏移，首先一定要记住是原图； 

　　此时读者肯定有疑问，原图哪里来的坐标呢？ 

　　这里我要解答最后两个问题了： 

　　首先我们知道有H x W个结果，我们随机取一点，它跟原图肯定是有个一一映射关系的，由于原图和特征图大小不同，所以特征图上的一个点对应原图肯定是一个框，然而这个框很小，比如说8 x 8，这里8是指原图和特征图的比例，所以这个并不是我们想要的框，那我们不妨把框的左上角或者框的中心作为锚点（Anchor），然后想象出一堆框，具体多少，聪明的读者肯定已经猜到，K个，这也就是图中所说的K anchor boxes（由锚点产生的K个框）；换句话说，H x W个点，每个点对应原图有K个框，那么就有H x W x k个框默默的在原图上，那RPN的结果其实就是判断这些框是不是物体以及他们的偏移；那么K个框到底有多大，长宽比是多少？这里是预先设定好的，共有9种组合，所以k等于9，最后我们的结果是针对这9种组合的，所以有H x W x 9个结果，也就是18个分数和36个坐标； 



![img](https://img2018.cnblogs.com/blog/1414369/201903/1414369-20190323160957616-1225858473.png)



**3. RPN的整个流程回顾**



　　最后我们再把RPN整个流程走一遍，首先通过一系列卷积得到公共特征图，假设他的大小是N x 16 x 16，然后我们进入RPN阶段，首先经过一个3 x 3的卷积，得到一个256 x 16 x 16的特征图，也可以看作16 x 16个256维特征向量，然后经过两次1 x 1的卷积，分别得到一个18 x 16 x 16的特征图，和一个36 x 16 x 16的特征图，也就是16 x 16 x 9个结果，每个结果包含2个分数和4个坐标，再结合预先定义的Anchors，经过后处理，就得到候选框；整个流程如图5：





![img](https://img2018.cnblogs.com/blog/1414369/201903/1414369-20190323161326543-1091383077.png)                                **图5 RPN整个流程**