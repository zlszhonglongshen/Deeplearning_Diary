* [Batch Normalization的诅咒](https://mp.weixin.qq.com/s/EtIt1gdg99f0mPr7wRU0Sg)

### 什么是BN？

在训练过程中，当我们更新之前的权重时，每个中间激活层的输出分布会在每次迭代时发生变化，这种现象称为内部协变量移位（ICS），所以很自然的一件事，如果我们想防止这种情况发生，就是修正所有的分布。简单的来说，如果分布变化了，会限制这个分布，不让他移动，以帮助梯度优化和防止梯度消失，这将帮助我的神经网络训练更快，因此减少这种内部协变量移位是推动BN发展的关键。

### 它是如何工作？

Batch Normalization通过在batch上减去经验平均值除以经验标准差来对前一个输出层的输出进行归一化。这将使数据看起来像**高斯分布**。

![img](https://mmbiz.qpic.cn/mmbiz_png/KYSDTmOVZvrs1K6tQU0cI5pVoQJySe10m8ondLjVTGxKUFvapVw6dZvtPwB5aw2icC3wCo4MqOUDz7ypvcP2OHA/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

其中*μ*和*σ^2^*分别为批均值和批方差。

![img](https://mmbiz.qpic.cn/mmbiz_png/KYSDTmOVZvrs1K6tQU0cI5pVoQJySe10m8ondLjVTGxKUFvapVw6dZvtPwB5aw2icC3wCo4MqOUDz7ypvcP2OHA/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

并且，我们学习了一个新的平均值和协方差*γ*和*β*。所以，简而言之，你可以认为batch normalization是帮助你控制batch分布的一阶和二阶动量。

![img](https://mmbiz.qpic.cn/mmbiz_png/KYSDTmOVZvrs1K6tQU0cI5pVoQJySe10Nl6ts7hzqTqyuDF5L1CN9sTcod8WDmpxZPqedsfgict9iaibULmRSRk4Q/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

vgg16网络的中间卷积层的特征分布输出。(左)没有任何归一化，(右)应用了batch normalization

### 优点

我将列举使用BN的一些好处

* 更快的收敛

* 降低初始化权重的重要性

* 鲁棒的超参数

* 需要减少的数据进行泛化

  ![img](https://mmbiz.qpic.cn/mmbiz_png/KYSDTmOVZvrs1K6tQU0cI5pVoQJySe10WK9q2rLv7NOBkOLsHWEbzclGhHjL6t3XPJJqRkEe3ogA1X43miaJaPw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

### 在使用小batch size的时候不稳定

如上所述，BN必须计算平均值和方差，以便batch中对之前的输出进行归一化。如果batch大小比较大的话，这种统计估计是比较准确的，而随着batch大小的减少，估计的准确性持续缩小。

![img](https://mmbiz.qpic.cn/mmbiz_png/KYSDTmOVZvrs1K6tQU0cI5pVoQJySe10Cz9mWF9bLOTpGD7mYxG0xCaUhtPDTNd2zFdDKQWkR7rVIwGfp85uTw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

以上是ResNet-50的验证错误图。可以推断，如果batch大小保持为32，它的最终验证误差在23左右，并且随着batch大小的减小，误差会继续减小(batch大小不能为1，因为它本身就是平均值)。损失有很大的不同(大约10%)。

如果batch大小是一个问题，为什么我们不使用更大的batch？我们不能在每种情况下都使用更大的batch。在finetune的时候，我们不能使用大的batch，以免过高的梯度对模型造成伤害。在分布式训练的时候，大的batch最终将作为一组小batch分布在各个实例中。