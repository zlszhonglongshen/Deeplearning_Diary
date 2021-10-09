# deeplearning-questions



主要针对面试中遇到的常见问题进行总结以及简略回答。

##  目标检测

### 1.anchor具体坐标方法，并简述它在faster-RCNN，V3，SSD有什么区别

https://zhuanlan.zhihu.com/p/231168560)



## 3.目标检测中正负样本比例严重失衡问题

### 第一种方法，focal LOSS

**focal loss原理**

1.focal loss 主要是为了解决one-stage目标检测中正负样本比例严重失衡的问题，该损失函数降低了大量简单负样本在训练中所占的权重，也可以理解为一种困难样本挖掘

2.损失函数的形式

focal loss是在交叉损失函数的基础上进行的修改，首先回顾二分类交叉上损失：

![交叉损失函数](https://gitee.com/zhonglongshen/questions/blob/master/images/1055519-20180818162755861-24998254.png)

y’是经过激活函数的输出，所以在0-1之间，可见普通的交叉熵对于正样本而言，输出概率越大损失越小。对于负样本而言，输出概率越小则损失越小。此时损失函数在大量简单样本名的迭代过程

中比较缓慢且可能无法优化到最优。那么focal loss是怎么改进的呢？

![img](https://gitee.com/zhonglongshen/questions/blob/master/images/focal_loss.png)

![img](https://gitee.com/zhonglongshen/questions/blob/master/images/focal-loss-2.png)

首先在原有的基础上加了一个因子，其中gamma>0使得减少易分类样本的损失，使得更关注与困难的、错分的样本。

例如gamma为2，对于正样本而言，预测结果为0.95肯定是简单简单样本，所以（1-0.95）的gamma次方就会很小，这时损失函数值就变得更小。对于负类样本而言同样，预测0.1的结果应当比预测0.7的样本损失值要小很多。对于预测概率为0.5时，

损失只减少0.25倍，所以更加关注与这种难以区分的样本。这样减少了简单样本的影响，大量预测概率很小的样本叠加起来后的效应才可能比较有效。

此时，加入平衡因子alpha，用来平衡正负样本本身的比例不均。文中alpha取0.25，即正样本要比负样本占比小，这是因为负类样本容易区分。

![img](https://gitee.com/zhonglongshen/questions/blob/master/images/focal-loss-3.png)

只添加alpha虽然可以平衡正负样本的重要性，但是无法解决简单与困难样本的问题。

gamma调节简单样本权重降低的速率，当gamma为-时，即为交叉熵损失函数，当gamma增加时，调整因子的影响也在增加，试验返现gamma为2位最优。

3.总结

作者认为one-stage和two-stage的表现差异主要原因是大量前背景类别不平衡导致。作者设计了一个简单密集型网络retinanet来训练在保证速度的同时达到了精度最优。

在双阶段算法中，在候选框阶段，通过得分和nms帅选掉了大量的负样本，然后在分类回归阶段又固定了正负样本比例。而one-stage阶段需要产生100k的候选位置，虽然有类似的采样，但是训练仍然被大量负样本所主导。

### 文章参考

[Focal loss论文详解](https://zhuanlan.zhihu.com/p/49981234)

### LOSS NAN ?

\* 出现loss nan的情况尽量设置大一点的warm-up-epoch的值，或者小一点的学习率，多试几次。如果使用的是one-stage的训练过程，使用adam优化器可能会出现nan的问题，请选择momentum optimizer。

## 