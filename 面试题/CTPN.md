# 场景文字检测-CTPN原理以及实现



对于复杂场景的文字识别，首先要定位文字的位置，即文字检测，这一直是一个研究热点。



[Detecting Text in Natural Image with Connectionist Text Proposal Networkarxiv.org](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1609.03605)



CTPN结合CNN与LSTM深度网络，能有效的检测出复杂场景的横向文字，效果如图1，是目标比较好的文字检测，由于CTPN是从faster rcnn改进来的，要对cnn原理和faster-rcnn网路结构有所了解。![img](https://pic1.zhimg.com/80/v2-658b7caad2c9a8163efe314e7cd6f284_720w.jpg)



## CTPN网络结构



原始CTPN只检测横向排列的文字，CTPN结构与faster RCNN基本类似，但是加入了LSTM层，假设输入**N**

images：

* 首先VGG提取特征，获得大小为NxCxHxW的conv5 feature map。

* 之后conv5上做3x3的滑动窗口，即每个点都结合周围3x3区域特征获得一个长度为3x3xC的特征向量。输出Nx9CxHxW的feature map，该特征显然只有CNN学习到的空间特征。

*  再讲这个feature map进行Reshape

  ![img](https://pic3.zhimg.com/80/v2-74fdcc9eeee49b7fb73acb4ba8aabef2_720w.png)

* 然后再以Batch=NH 且最大时间长度Tmax = W 的数据流输入双向LSTM，学习每一行的序列特征。双向LSTM输出(NxH)xWx256,再以Reshape恢复形状：

  ![img](https://pic2.zhimg.com/80/v2-27b8e73c01bf7962c53589f78b4aba6e_720w.png)

该特征即包含空间特征，也包含了LSTM学习到的序列特征。

* 然后经过“FC”卷积层，变为Nx512xHxW的特征

* 最后经过类似Faster-RCNN的RPN网络，获得text proposals，如图



  ![img](https://pic2.zhimg.com/80/v2-b29f366f73ac0fba695435770e85809e_720w.jpg)



这里讲一下conv5 feature map如何从NxCxHxW 变为 Nx9CxHxW：



![img](https://picb.zhimg.com/80/v2-4399a8ecb012241fa542e084eb7d727f_720w.jpg)



在原版caffe代码中使用im2col提取每个点附近的9点临近点，然后每行都如何处理：

![img](https://picb.zhimg.com/80/v2-2eea478f055fbf7aea37cb1d11119ced_720w.png)

接着每个通道都如何处理：

![img](https://pic3.zhimg.com/80/v2-8c07ab22c9c50c68b907db7d79c44f40_720w.png)

而im2col是用于卷积加速的操作，即将卷积变为矩阵乘法，从而使用blas库快速计算。

***特别说明：上述是对原paper+caffe代码的解释，其他代码实现异同不在本文讨论范围内***

接下来，文章围绕下面三个问题展开：

1：为何使用双向lstm

2：如何通过fc层输出产生图2-b中的Text proposals

3：如何通过Text proposals确定最终的文本位置，即文本线构造算法

## 为何使用双向LSTM

- 对于RNN原理不了解的读者，请先参考RNN原理介绍：

[完全解析RNN，Seq2Seq和Attention机制zhuanlan.zhihu.com![图标](https://pic3.zhimg.com/v2-f74c69bc39f8d8516e22d88ef647c3a0_180x120.jpg)](https://zhuanlan.zhihu.com/p/51383402)



- [关于LSTM长短期记忆模型，请参考](https://link.zhihu.com/?target=http%3A//colah.github.io/posts/2015-08-Understanding-LSTMs/)

- CTPN中为何使用双向LSTM？

![img](https://pic4.zhimg.com/80/v2-8d72777321cbf1336b79d839b6c7f9fc_720w.jpg)

CNN学习的是感受野内的空间信息，LSTM学习的是序列特征。对于文本序列检测，显然既需要CNN抽象空间特征，也需要序列特征（毕竟文字是连续的）；

CTPN中使用双向LSTM，相比一般单向的LSTM有什么优势？双向lstm实际上就是将2个方向相反的lstm连起来。

![img](https://pic1.zhimg.com/80/v2-bc5266c4587af49516adb2cee4351838_720w.jpg)



一般说，双向lstm都好于单向lstm。还是看lstm介绍文章中的例子：

\> 我的手机坏了，我打算一部新手机。

假设使用LSTM对空白部分填词。如果只看横线前面的词，“手机坏了”，那么“我”是打算“修”还是“买”还是“大哭一场”？双向LSTM能看到后面的词是“一部新手机“，那么横线上的词填“买“的概率就大得多了。显然对于文字检测，这种情况也依然适用。

## 如何通过“FC”卷积层输出产生图2-b中的Text proposals?



![img](https://pic2.zhimg.com/80/v2-8496528d21dfd1c4e90df4ff57fa6221_720w.jpg)







CTPN通过CNN和BLSTM学到一组“空间+序列”特征后，在“FC”卷积层后接入RPN网络。这里的RPN与Faster-RCNN类似，分为两个分支：

1：左边分支用于bounding box regression。由于FC feature map每个点都配备了10个anchor，同时只回归中心y坐标与高度2个值，所以rpn_bboxp_red有20个channels

2：右边分支用于softmax分类Anchor。

具体RPN网络与Faster RCNN完全一样，所以不在介绍，只分析不同之处。

## 竖直Anchor定位文字位置

由于CTPN针对的是横向排列的文字检测，所以其采用一组（10个）等宽度的anchors，用于定位文字位置。Anchor宽高为：



![img](https://pic4.zhimg.com/80/v2-0d777cb27dbb89bf925ca9d90211383d_720w.png)

需要注意，由于CTPN采用VGG16模型提取特征，那么conv5 feature map的宽高都是输入Image的宽高的1/16,同时FC与conv width和height都相等。

如图所示，CTPN为FC feature map每一个点都配备10个上述Anchors。



![img](https://pic2.zhimg.com/80/v2-93e22f54fb0231b3f763f2f8129913ad_720w.jpg)



这样设置anchor是为了：

1：保证在x方向上，Anchor覆盖原图每个点且不互相重叠

2：不同文本在y方向上高度差距很大，所以设置Anchor高度为11-283，用于覆盖不同高度的文本目标

多说一句，

多说一句，我看还有人不停的问Anchor大小为什么对应原图尺度，而不是conv5/fc特征尺度。这是因为Anchor是目标的候选框，经过后续分类+位置修正获得目标在原图尺度的检测框。那么这就要求Anchor必须是对应原图尺度！除此之外，如果Anchor大小对应conv5/fc尺度，那就要求Bounding box regression把很小的框回归到很大，这已经超出Regression小范围修正框的设计目的。

获得Anchor后，与Faster R-CNN类似，CTPN会做如下处理：

1. Softmax判断Anchor中是否包含文本，即选出Softmax score大的正Anchor

2. Bounding box regression修正包含文本的Anchor的***\*中心y坐标\****与***\*高度\****。

注意，与Faster R-CNN不同的是，这里Bounding box regression不修正Anchor中心x坐标和宽度。具体回归方式如下：



![img](https://pic1.zhimg.com/80/v2-738d5b097b64f8012cef7b9d3c05f7b2_720w.jpg)



其中， ![[公式]](https://www.zhihu.com/equation?tex=v%3D%28v_c%2C+v_h%29) 是回归预测的坐标， ![[公式]](https://www.zhihu.com/equation?tex=v%3D%28v_c%5E%2A%2C+v_h%5E%2A%29) 是Ground Truth， ![[公式]](https://www.zhihu.com/equation?tex=c_y%5Ea) 和 ![[公式]](https://www.zhihu.com/equation?tex=h%5Ea) 是Anchor的中心y坐标和高度。Bounding box regression具体原理请参考之前文章。

Anchor经过上述Softmax和 ![[公式]](https://www.zhihu.com/equation?tex=y) 方向bounding box regeression处理后，会获得图7所示的一组竖直条状text proposal。后续只需要将这些text proposal用文本线构造算法连接在一起即可获得文本位置。



![img](https://pic3.zhimg.com/80/v2-447461eb54bcc3c93992ffd1c70bcfb8_720w.jpg)



在论文中，作者也给出了直接使用Faster R-CNN RPN生成普通proposal与CTPN LSTM+竖直Anchor生成text proposal的对比，如图8，明显可以看到CTPN这种方法更适合文字检测。



![img](https://pic2.zhimg.com/80/v2-82a34bf3b3591c4a21d90e8997ed1534_720w.jpg)



##  文本线构造算法

在一个步骤中，已经获得了图所示的一串或者多串text proposals，接下来就要采用文本线构造办法，把浙西诶text proposals链接成一个文本检测框。



![img](https://pic3.zhimg.com/80/v2-de8098e725d168a038f197ce0707faaf_720w.jpg)



为了说名问题，假设某张图有图9所以的2个text proposals。即蓝色和红色2组Anchor，CTPN采用如下算法构造文本线：

1. 按照水平 ![[公式]](https://www.zhihu.com/equation?tex=x) 坐标排序Anchor

2. 按照规则依次计算每个Anchor ![[公式]](https://www.zhihu.com/equation?tex=%5Ctext%7Bbox%7D_i) 的 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctext%7Bpair%7D%28%5Ctext%7Bbox%7D_j%29) ，组成 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctext%7Bpair%7D%28%5Ctext%7Bbox%7D_i%2C+%5Ctext%7Bbox%7D_j%29)

3. 通过 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctext%7Bpair%7D%28%5Ctext%7Bbox%7D_i%2C+%5Ctext%7Bbox%7D_j%29) 建立一个Connect graph，最终获得文本检测框

   下面详细解释，假设每个Anchor index如绿色数字，同时每个Anchor softmax score如黑色数字。

***\*文本线构造算法通过如下方式建立每个Anchor\**** ![[公式]](https://www.zhihu.com/equation?tex=%5Ctext%7Bbox%7D_i) ***\*的\**** ![[公式]](https://www.zhihu.com/equation?tex=%5Ctext%7Bpair%7D%28%5Ctext%7Bbox%7D_i%2C+%5Ctext%7Bbox%7D_j%29) ***\*：\****

正向寻找：

1. 沿水平正方向，寻找和 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctext%7Bbox%7D_i) 水平距离小于50像素的候选Anchor（每个Anchor宽16像素，也就是最多正向找50/16=3个）

2. 从候选Anchor中，挑出与 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctext%7Bbox%7D_i) 竖直方向 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctext%7Boverlap%7D_v+%3E0.7) 的Anchor

3. 挑出符合条件2中Softmax score最大的 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctext%7Bbox%7D_j)

   再反向寻找：

1. 沿水平负方向，寻找和 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctext%7Bbox%7D_j) 水平距离小于50的候选Anchor

2. 从候选Anchor中，挑出与 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctext%7Bbox%7D_j) 竖直方向 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctext%7Boverlap%7D_v+%3E0.7) 的Anchor

3.  挑出符合条件2中Softmax score最大的 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctext%7Bbox%7D_k)

   最后对比 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctext%7Bscore%7D_i) 和 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctext%7Bscore%7D_k) :

1. 如果 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctext%7Bscore%7D_i+%3E%3D+%5Ctext%7Bscore%7D_k) ，则这是一个最长连接，那么设置 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctext%7BGraph%7D%28i%2C+j%29+%3D+%5Ctext%7BTrue%7D)

  2. 如果 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctext%7Bscore%7D_i+%3C+%5Ctext%7Bscore%7D_k) ，说明这不是一个最长的连接（即该连接肯定包含在另外一个更长的连接中）。



\```python

lass TextProposalGraphBuilder:

​    .....

​    def is_succession_node(self, index, succession_index):

​        precursors=self.get_precursors(succession_index)

​        \# index 为上文中的 i, succession_index 为 j, precursors 为负向搜索找到的 k

​        if self.scores[index]>=np.max(self.scores[precursors]):

​            return True

​        return False



​    def build_graph(self, text_proposals, scores, im_size):

​        self.text_proposals=text_proposals

​        self.scores=scores

​        self.im_size=im_size

​        self.heights=text_proposals[:, 3]-text_proposals[:, 1]+1



​        boxes_table=[[] for _ in range(self.im_size[1])]

​        for index, box in enumerate(text_proposals):

​            boxes_table[int(box[0])].append(index)

​        self.boxes_table=boxes_table



​        graph=np.zeros((text_proposals.shape[0], text_proposals.shape[0]), np.bool)



​        for index, box in enumerate(text_proposals):

​            \# 沿水平正方向寻找所有overlap_v > 0.7匹配

​            successions=self.get_successions(index)

​            if len(successions)==0:

​                continue



​            \# 找匹配中socre最大的succession_index（即上文j）

​            succession_index=successions[np.argmax(scores[successions])] 



​            \# 沿水平负方向寻找socre最大的 k，如果socre_i >= score_k 则是一个最长连接

​            if self.is_succession_node(index, succession_index):

​                \# NOTE: a box can have multiple successions(precursors) if multiple successions(precursors)

​                \# have equal scores.

​                \# 设置 Graph connection (i,j)为 True

​                graph[index, succession_index]=True

​        return Graph(graph)

\```



![img](https://picb.zhimg.com/80/v2-822f0709d3e30df470a8e17f09a25de0_720w.jpg)



举例说明，如图10，Anchor已经按照 ![[公式]](https://www.zhihu.com/equation?tex=x) 顺序排列好，并具有图中的Softmax score（这里的score是随便给出的，只用于说明文本线构造算法）：



\1. 对 ![[公式]](https://www.zhihu.com/equation?tex=i%3D3) 的 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctext%7Bbox%7D_3) ，向前寻找50像素，满足 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctext%7Boverlap%7D_v%3E0.7) 且score最大的是 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctext%7Bbox%7D_7) ，即 ![[公式]](https://www.zhihu.com/equation?tex=j%3D7) ； ![[公式]](https://www.zhihu.com/equation?tex=%5Ctext%7Bbox%7D_7) 反向寻找，满足 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctext%7Boverlap%7D_v+%3E0.7) 且score最大的是 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctext%7Bbox%7D_3) ，即 ![[公式]](https://www.zhihu.com/equation?tex=k%3D3) 。由于 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctext%7Bscore%7D_3+%3E%3D+%5Ctext%7Bscore%7D_3) ， ![[公式]](https://www.zhihu.com/equation?tex=%5Ctext%7Bpair%7D%28%5Ctext%7Bbox%7D_3%2C+%5Ctext%7Bbox%7D_7%29) 是最长连接，那么设置 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctext%7BGraph%7D%283%2C7%29+%3D+%5Ctext%7BTrue%7D)

\2. 对 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctext%7Bbox%7D_4) 正向寻找得到 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctext%7Bbox%7D_7) ； ![[公式]](https://www.zhihu.com/equation?tex=%5Ctext%7Bbox%7D_7) 反向寻找得到 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctext%7Bbox%7D_3) ，但是 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctext%7Bscore%7D_4+%3C+%5Ctext%7Bscore%7D_3)，即 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctext%7Bpair%7D%28%5Ctext%7Bbox%7D_4%2C+%5Ctext%7Bbox%7D_7%29) 不是最长连接，包含在 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctext%7Bpair%7D%28%5Ctext%7Bbox%7D_3%2C+%5Ctext%7Bbox%7D_7%29) 中。



***\*然后，这样就建立了一个\**** ![[公式]](https://www.zhihu.com/equation?tex=N%5Ctimes+N) ***\*的Connect graph（其中\**** ![[公式]](https://www.zhihu.com/equation?tex=N) ***\*是正Anchor数量）。遍历Graph：\****



\1. ![[公式]](https://www.zhihu.com/equation?tex=%5Ctext%7BGraph%7D%280%2C3%29+%3D+%5Ctext%7BTrue%7D) 且 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctext%7BGraph%7D%283%2C7%29+%3D+%5Ctext%7BTrue%7D) ，所以Anchor index 0->3->7组成一个文本，即蓝色文本区域。

\2. ![[公式]](https://www.zhihu.com/equation?tex=%5Ctext%7BGraph%7D%286%2C10%29+%3D+%5Ctext%7BTrue%7D) 且 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctext%7BGraph%7D%2810%2C12%29+%3D+%5Ctext%7BTrue%7D) ，所以Anchor index 6->10->12组成另外一个文本，即红色文本区域。



这样就通过Text proposals确定了文本检测框。



## 训练策略



由于作者没有给出CTPN原始训练代码，所以此处仅能根据论文分析。

![img](https://pic3.zhimg.com/80/v2-06f4af15265c1a37269d0b9919daa79b_720w.png)



明显可以看出，该Loss分为3个部分：

\1. Anchor Softmax loss：该Loss用于监督学习每个Anchor中是否包含文本。 ![[公式]](https://www.zhihu.com/equation?tex=s_i%5E%2A%3D%5C%7B0%2C1%5C%7D) 表示是否是Groud truth。

\2. Anchor y coord regression loss：该Loss用于监督学习每个包含为本的Anchor的Bouding box regression y方向offset，类似于Smooth L1 loss。其中 ![[公式]](https://www.zhihu.com/equation?tex=v_j) 是 ![[公式]](https://www.zhihu.com/equation?tex=s_i) 中判定为有文本的Anchor，或者与Groud truth vertical IoU>0.5。

\3. Anchor x coord regression loss：该Loss用于监督学习每个包含文本的Anchor的Bouding box regression x方向offset，与y方向同理。前两个Loss存在的必要性很明确，但这个Loss有何作用作者没有解释（从训练和测试的实际效果看，作用不大）

说明一下，在Bounding box regression的训练过程中，其实只需要注意被判定成正的Anchor，不需要去关心杂乱的负Anchor。这与Faster R-CNN类似。



## 总结

\1. 由于加入LSTM，所以CTPN对水平文字检测效果超级好。

\2. 因为Anchor设定的原因，CTPN只能检测横向分布的文字，小幅改进加入水平Anchor即可检测竖直文字。但是由于框架限定，对不规则倾斜文字检测效果非常一般。

\3. CTPN加入了双向LSTM学习文字的序列特征，有利于文字检测。但是引入LSTM后，在训练时很容易梯度爆炸，需要小心处理。



# 自然场景文本检测技术综述（CTPN，EAST）

##  ctpn如何把这些小文本框准确的检测出来呢？

CTPN借助Faster RCNN中的anchor机制，使用RPN能有效的用单一尺寸的滑动窗口来检测多尺寸的物体。当然CRPN根据文本检测的特点做了比较多的创新，比如CRPN的anchor机制是直接回归预测物体的四个参数（x,y,w,h)，但是CRPN采取了回归两个参数(y,h)，即anchor的纵向偏移以及该anchor的文本框的高度，因为每个候选框的宽度w已经规定为16个像素，不需要再学习，而x坐标直接使用anchor的x坐标，也不用学习，所以CTPN的思路就是只学习y和h这两个参数来完成小候选框的检测~跟RPN类似，CTPN中对于每个候选框都是用了K个不同的anchors（k在这里默认是10），但是与RPN不同的是，这里的anchors的width是固定的16个像素，而height的高度范围为11-273（每次对输入图像的height除以0.7，一共K个高度），当然CTPN中还是保留了RPN大多数的思路，比如还是要预测候选框的分数score（该候选框有文本和无文本的得分）。



这么多小尺度候选框怎么才能串联成一个完成的文本行呢？



文本行构建很简单，通过将那些text/no-text score>0.7 的连续的text proposals相连接即可，文本行的构建如下，首先，为一个proposal BI定义一个邻居（Bj)：Bj>Bi，其中：



1：Bj在水平距离上离Bi最近



2：该距离小于50 pixels



3：它们的垂直重叠(vertical overlap)>0.7



另外，如果同时满足Bj−>Bi和Bi−>Bj，会将两个proposals被聚集成一个pair。接着，一个文本行会通过连续将具有相同proposal的pairs来进行连接来构建。



接下来我们就较为细节地学习一下这个CTPN经典网络。





![img](https://img2018.cnblogs.com/blog/1093303/201810/1093303-20181012095613827-270550342.png)



首先CTPN的基础网络使用了VGG16用于特征提取，在VGG的最后一个卷积层Conv5，CTPN用了3x3的卷积核来对该feature map做卷积，这个conv5特征图的尺寸由输入图像来决定，而卷积时的步长却限定为16，感受野被固定为228个像素，卷积后的特征被送入BLSTM继续学习，最后街上一层全连接层FC输出我们预测的参数：2K个纵向坐标y，2k个分数，k个x的水平偏移量。看到这里大家可能有个疑问，这个x的偏移到底是什么，为什么需要回归这个参数？如果需要X的参数，为什么不在候选框参数回归时直接预测成（x,y,h）三个参数呢，而要多此一举把该参数单独预测？这个X的作用作者提到这也是他们论文的一大亮点，称之为Side-refinement，我理解为文本框边缘优化。我们回顾一下上面提到的一个问题，文本框检测中边缘部分的预测并不准确。那么改咋办，CTPN就是用这个X的偏移量来精修边缘问题。这个X是指文本框在水平方向的左边界和右边界，我们通过回归这个左边界和右边界参数进而可以使得我们对文本框的检测更为精准。在这里想举个例子说明一下回归这个x参数的重要性。



我们观察下图，第一幅图张我们看到我们有很多小候选框，位于左边的候选框我标记为1、2、3、4号框,1号框和2号框为蓝色，表明得分不高我们不把这两个框合并到大文本框内，对于3号框和4号框那就比较尴尬了，如果取3号框作为文本框的边缘框，那么显然左边边缘留白太多，精准度不够，但如果去掉3号框而使用4号框作为左边缘框，则有些字体区域没有检测出来，同样检测精度不足。这种情况其实非常容易出现，所以CTPN采取了Side-refinement 思路进一步优化边缘位置的预测即引入回归X参数，X参数直接标定了完整文本框的左右边界，做到精确的边界预测。第二幅图中的红色框就是经过Side-refinement后的检测结果，可以看出检测准确率有了很大的提升。 side-refinement确实可以进一步提升位置准确率，在SWT的Multi-Lingual datasets上产生2%的效果提升。



![img](https://img2018.cnblogs.com/blog/1093303/201810/1093303-20181012095628313-1925279983.png)



再看多几幅图，体验一下Side-refinement后的效果。



![img](https://img2018.cnblogs.com/blog/1093303/201810/1093303-20181012095641132-90730013.png)



最后总结一下CTPN这个流行的文本检测框架的三个闪光点：



\- 将文本检测任务转化为一连串小尺度文本框的检测；

\- 引入RNN提升文本检测效果；

\- Side-refinement（边界优化）提升文本框边界预测精准度。



![img](https://img2018.cnblogs.com/blog/1093303/201810/1093303-20181012095807274-1282513018.png)



当然，CTPN也有一个很明显的缺点：对于非水平的文本的检测效果并不好。CTPN论文中给出的文本检测效果图都是文本位于水平方向的，显然CTPN并没有针对多方向的文本检测有深入的探讨。那对于任意角度的文本检测应该采取什么的算法思路呢？下面的SegLink算法给出了一个新奇的解决方案。





\# CTPN讲解3

\## 算法详解

\## 1. 算法流程

CTPN的流程和Faster R-CNN的RPN网络类似，首先使用VGG-16提取特征，在conv5进行 ![[公式]](https://www.zhihu.com/equation?tex=3%5Ctimes3) ，步长为1的滑窗。设conv5的尺寸是 ![[公式]](https://www.zhihu.com/equation?tex=W%5Ctimes+H) ，这样在conv5的同一行，我们可以得到W个256维的特征向量。将同一行的特征向量输入一个双向LSTM中，在双向LSTM后接一个512维的全连接后便是CTPN的3个任务的多任务损失，结构如图1。任务1的输出是 ![[公式]](https://www.zhihu.com/equation?tex=2%5Ctimes+k) ，用于预测候选区域的起始 ![[公式]](https://www.zhihu.com/equation?tex=y) 坐标和高度 ![[公式]](https://www.zhihu.com/equation?tex=h) ；任务2是用来对前景和背景两个任务的分类评分；任务3是 ![[公式]](https://www.zhihu.com/equation?tex=k)个输出的side-refinement的偏移（offset）预测。在CTPN中，任务1和任务2是完全并行的任务，而任务3要用到任务1，2的结果，所以理论上任务3和其他两个是串行的任务关系，但三者放在同一个损失和函数中共同训练，也就是我们在Faster R-CNN中介绍的近似联合训练。



![img](https://pic3.zhimg.com/80/v2-1fc149812bc5ff7973b4705b25634bef_720w.jpg)图1：CTPN的结构



\------



\## 2. 数据准备



和RPN的要求一样，CTPN输入图像的尺寸无硬性要求，只是为了保证特征提取的有效性，在保证图片比例不变的情况下，CTPN将输入图片的resize到600，且保证长边不大于1000。



\```python

def resize_im(im, scale, max_scale=None):

​    f=float(scale)/min(im.shape[0], im.shape[1])

​    if max_scale!=None and f*max(im.shape[0], im.shape[1])>max_scale:

​        f=float(max_scale)/max(im.shape[0], im.shape[1])

​    return cv2.resize(im, (0, 0), fx=f, fy=f), f

\```



\------



## CTPN的锚点机制



作者通过分析RPN在场景文字检测的实验发现RPN的效果并不是特别理想，尤其是在定位文本区域的横坐标上存在很大的误差。因为在一串文本中，在不考虑语义的情况下，每个字符都是一个独立的个体，这使得文字区域的边界是很难确定的。显然，文本区域检测和物体检测最大的区别是文本区域是一个序列，如图2。如何我们能根据文本的序列特征捕捉到文本区域的边界信息，应该能够对本文区域的边界识别能够很好的预测。

![img](https://picb.zhimg.com/80/v2-9e698d277aadf0d2a18049fe02b51a3f_720w.jpg)图2：文本区域的序列特征



而在目前的神经网络中，RNN在处理序列数据上占有垄断性的优势地位。在RNN的训练过程中，数据是以时间片为单位输入到模型中的。所以，如何将文本区域变成可以序列化输入的顺序成为了CTPN一个重要的要求。如图2所展示的，每一个蓝色矩形是一个锚点，那么一个文本区域便是由一系列宽度固定，紧密相连的锚点构成。所以，CTPN有如下的锚点设计机制：



由于CTPN是使用的VGG-16进行特征提取，VGG-16经过4次max pooling的降采样，得到的_feature_stride=16，_feature_stride体现在在conv5上步长为1的滑窗相当于在输入图像上步长为16的滑窗。所以，根据VGG-16的网络结构，CTPN的锚点宽度w必须为16。对于一个输入序列中的所有锚点，如果我们能够判断出锚点的正负，把这一排正锚点连在一起便构成了文本区域，因此锚点的起始坐标 ![[公式]](https://www.zhihu.com/equation?tex=x) 也不用预测。所以在CTPN中，网络只需要预测锚点的起始 ![[公式]](https://www.zhihu.com/equation?tex=y) 坐标以及锚点的高度 ![[公式]](https://www.zhihu.com/equation?tex=h) 即可。



在RPN网络中，一个特征向量对应的多个尺寸和比例的锚点，同样的，CTPN也对同一个特征向量设计了10个锚点。在CTPN中，锚点的高度依次是[11,16,23,33,48,68,97,139,198,283]，即高度每次除以0.7。



\------



\## 4. CTPN中的RNN



我们多次强调场景文字检测一个重要的不同是文本区域具有序列特征。在上面一段，我们已经可以根据锚点构造序列化的数据。通过在 ![[公式]](https://www.zhihu.com/equation?tex=W%5Ctimes+H) 的conv5层进行步长为的滑窗，每一次横向滑动得到的便是W个长度为256的特征向量 ![[公式]](https://www.zhihu.com/equation?tex=X_t) 。设RNN的隐层节点是 ![[公式]](https://www.zhihu.com/equation?tex=H_t) ，则RNN模型可以表示为



![[公式]](https://www.zhihu.com/equation?tex=H_t+%3D+%5Cvarphi%28H_t-1%2C+X_t%29%5Cquad+t+%3D+1%2C2%2C...%2CW+%5Ctag%7B1%7D)



其中$$\varphi$$是非线性的激活函数。隐层节点的数量是128，RNN使用的是双向的LSTM，因此通过双向LSTM得到的特征向量是256维的。



\```text

layer {

  name: "lstm"

  type: "Lstm"

  bottom: "lstm_input"

  top: "lstm"

  lstm_param {

​      num_output: 128

​      weight_filler {

​        type: "gaussian"

​        std: 0.01

​      }

​      bias_filler {

​        type: "constant"

​      }

​      clipping_threshold: 1

​    }

}



...



layer {

  name: "rlstm"

  type: "Lstm"

  bottom: "rlstm_input"

  top: "rlstm-output"

  lstm_param {

​    num_output: 128

   }

}



...



\# merge lstm and rlstm

layer {

  name: "merge_lstm_rlstm"

  type: "Concat"

  bottom: "lstm"

  bottom: "rlstm"

  top: "merge_lstm_rlstm"

  concat_param {

​    axis: 2

  }

}

\```



\------



\## 5. side-refinement



将side-refinement从CTPN独立出来似乎更好理解，side-refinement对于CTPN的其它部分相当于Faster R-CNN中的Fast R-CNN对于RPN。不同之处在于side-refinement根据CTPN预测的锚点信息优化文本行，而且side-refinement只优化文本行的横向坐标 ![[公式]](https://www.zhihu.com/equation?tex=x) 和宽度 ![[公式]](https://www.zhihu.com/equation?tex=h) ，在论文中这个信息叫做偏移（offset）。Fast R-CNN优化的是根据RPN的输出通过NMS得到的候选区域的位置四要素。所以，side-refinement一个重要的步骤是如何根据锚点信息构造文本行。



\## 5.1 文本行的构造



通过CTPN可以得到候选区域的的得分，如果判定为文本区域的得分大于阈值 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctheta_1) ，则该区域用来构造文本行。文本行是由一系列大于0.7的候选区域的***\*邻居对\****构成的，如果区域 ![[公式]](https://www.zhihu.com/equation?tex=B_j) 是区域 ![[公式]](https://www.zhihu.com/equation?tex=B_i)的邻居对，需要满足如下条件：



\1. ![[公式]](https://www.zhihu.com/equation?tex=B_j) 是距离 ![[公式]](https://www.zhihu.com/equation?tex=B_i) 最近的正文本区域；

\2. ![[公式]](https://www.zhihu.com/equation?tex=B_j) 和 ![[公式]](https://www.zhihu.com/equation?tex=B_i) 的距离小于 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctheta_2) 个像素值；

\3. ![[公式]](https://www.zhihu.com/equation?tex=B_i) 和 ![[公式]](https://www.zhihu.com/equation?tex=B_j) 的竖直方向的重合率大于 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctheta_3) 。



在源码提供的配置文件中![[公式]](https://www.zhihu.com/equation?tex=%5Ctheta_1%3D0.7%2C+%5Ctheta_2%3D50%2C+%5Ctheta_3%3D0.7)，文本区域是由一系列邻居对构成的。



\```python

def get_successions(self, index):

​    box=self.text_proposals[index]

​    results=[]

​    for left in range(int(box[0])+1, min(int(box[0])+cfg.MAX_HORIZONTAL_GAP+1, self.im_size[1])):

​        adj_box_indices=self.boxes_table[left]

​        for adj_box_index in adj_box_indices:

​            if self.meet_v_iou(adj_box_index, index):

​                results.append(adj_box_index)

​        if len(results)!=0:

​            return results

​    return results

\```



\## 5.2 side-refinement的损失函数



构造完文本行后，我们根据文本行的左端和右端两个锚点的特征向量计算文本行的相对位移 ![[公式]](https://www.zhihu.com/equation?tex=o) ：



![[公式]](https://www.zhihu.com/equation?tex=o+%3D+%28x_%7Bside%7D+-+c_x%5Ea%29%2Fw_a+%5Ctag%7B2%7D)



![[公式]](https://www.zhihu.com/equation?tex=o%5E%5Cstar+%3D+%28x%5E%7B%5Cstar%7D_%7Bside%7D-c%5Ea_x%29%2Fw_a+%5Ctag%7B3%7D)



其中 ![[公式]](https://www.zhihu.com/equation?tex=x*_%7Bside%7D) 是由CTPN构造的文本行的左侧和右侧两个锚点的 ![[公式]](https://www.zhihu.com/equation?tex=x) 坐标，即文本行的起始坐标和结尾坐标。所以 ![[公式]](https://www.zhihu.com/equation?tex=x%5E%7B%5Cstar%7D_*%7Bside%7D) 便是对应的ground truth的坐标， ![[公式]](https://www.zhihu.com/equation?tex=c_x%5Ea) 是锚点的中心点坐标， ![[公式]](https://www.zhihu.com/equation?tex=w_a) 是锚点的宽度，所以是16。side-refinement使用的损失函数是smooth L1函数。



\------



\## 6. CTPN的损失函数



CTPN使用的是Faster R-CNN的近似联合训练，即将分类，预测，side-refinement作为一个多任务的模型，这些任务的损失函数共同决定模型的调整方向。



\## 6.1 文本区域得分损失 ![[公式]](https://www.zhihu.com/equation?tex=L_s%5E%7Bcl%7D)



分类损失函数 ![[公式]](https://www.zhihu.com/equation?tex=L_s%5E%7Bcl%7D%28s_i%2Cs_i%5E%29) 是softmax损失函数，其中 ![[公式]](https://www.zhihu.com/equation?tex=s_i%5E%3D%5C%7B0%2C1%5C%7D) 是ground truth，即如果锚点为正锚点（前景）， ![[公式]](https://www.zhihu.com/equation?tex=s_i%5E%3D1) ，否则 ![[公式]](https://www.zhihu.com/equation?tex=s_i%5E%3D0) 。在tf的源码中(lib/rpn_msr/anchor_target_layer_tf.py)，一个锚点是正锚点的条件如下:



\1. 每个位置上的9个anchor中overlap最大的认为是前景；

\2. overlap大于0.7的认为是前景3



如果overlap小于0.3，则被判定为背景。在源码中参数RPN_CLOBBER_POSITIVES为true则表示如果一个样本的overlap小于0.3，且同时满足正样本的条件1，则该样本被判定为负样本。



![[公式]](https://www.zhihu.com/equation?tex=s_i) 是预测锚点 ![[公式]](https://www.zhihu.com/equation?tex=i) 为前景的概率。



\## 6.2 纵坐标损失 ![[公式]](https://www.zhihu.com/equation?tex=L_v%5E%7Bre%7D)



纵坐标的损失函数 ![[公式]](https://www.zhihu.com/equation?tex=L_v%5E%7Bre%7D%28v_j%2Cv_j%5E%5Cstar%29) **使用的是smooth l1 损失函数，** ![[公式]](https://www.zhihu.com/equation?tex=v_j) 和 ![[公式]](https://www.zhihu.com/equation?tex=v_h) 使用的是相对位移，表示如下



![[公式]](https://www.zhihu.com/equation?tex=v_c+%3D+%28c_y+-+c_y%5Ea%29%2Fh%5Ea+%5Ctag%7B3%7D)



![[公式]](https://www.zhihu.com/equation?tex=v_h+%3D+log%28h%2Fh%5Ea%29+%5Ctag%7B4%7D)



![[公式]](https://www.zhihu.com/equation?tex=v_c%5E+%3D+%28c_y%5E+-+c_y%5Ea%29%2Fh%5Ea+%5Ctag%7B5%7D)



![[公式]](https://www.zhihu.com/equation?tex=v_h%5E+%3D+log%28h%5E%2Fh%5Ea%29+%5Ctag%7B6%7D)



![[公式]](https://www.zhihu.com/equation?tex=v%3D%5C%7Bv_c%2C+v_h%5C%7D) **,** ![[公式]](https://www.zhihu.com/equation?tex=v%5E%5Cstar%3D%5C%7Bv%5E%5Cstar_c%2C+v%5E%7B%5Cstar%7D_h%5C%7D) 分别是预测的坐标和ground truth的坐标



综上，CTPN的损失函数表示为



![[公式]](https://www.zhihu.com/equation?tex=L%28s_i%2Cv_j%2Co_k%29+%3D+%5Cfrac%7B1%7D%7BN_s%7D%5Csum_i+L_s%5E%7Bcl%7D%28s_i%2Cs_i%5E%5Cstar%29+%2B+%5Cfrac%7B%5Clambda_1%7D%7BN_v%7D%5Csum_j+L_v%5E%7Bre%7D%28v_j%2Cv_j%5E%5Cstar%29+%2B+%5Cfrac%7B%5Clambda_2%7D%7BN_o%7DL%5E%7Bre%7D_o%28o_k%2Co_k%5E%5Cstar%29%5Ctag%7B7%7D+)



![[公式]](https://www.zhihu.com/equation?tex=%5Clambda_1) ， ![[公式]](https://www.zhihu.com/equation?tex=%5Clambda_2) 是各任务的权重系数， ![[公式]](https://www.zhihu.com/equation?tex=N_s) ， ![[公式]](https://www.zhihu.com/equation?tex=N_v) ， ![[公式]](https://www.zhihu.com/equation?tex=N_o) 是归一化参数，表示对应任务的样本数量。



\------

# CTPN的训练细节



每个minibatch同样采用“Image-centric”的采样方法，每次随机取一张图片，然后在这张图片中采样128个样本，并尽量保证正负样本数量的均衡。卷积层使用的是Image-Net上无监督训练得到的结果，权值初始化使用的是均值为0，标准差为0.01的高斯分布。SGD的参数中，遗忘因子是0.9，权值衰减系数是0.0006。前16k次迭代的学习率是0.001，后4k次迭代的学习率是0.0001。