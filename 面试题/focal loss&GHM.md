> https://zhuanlan.zhihu.com/p/80594704



### 第一部分 Focal Loss



Focal Loss的引入主要是为了解决**难易样本数量不平衡（注意，有区别于正负样本数量不平衡）**的问题，实际可以使用的范围非常广泛，为了简单方便解释，还是拿目标检测的应用场景来说明：

单阶段的目标检测器通常会产生高达100K的候选目标，只有极少数是正样本，正负样本数量极其不平衡。我们在计算分类的时候常用的损失-交叉熵的公式如下：

![[公式]](https://www.zhihu.com/equation?tex=CE%3D%5Cleft%5C%7B+%5Cbegin%7Baligned%7D+-log%28p%29++%2C%5Cquad+if%5Cquad++y%3D1%5C%5C+-log%281-p%29%2C%5Cquad+if%5Cquad++y%3D0++++%5Cend%7Baligned%7D+%5Cright.)

为了**解决正负样本不平衡**的问题，我们通常会在交叉损失的前面加上一个参数![[公式]](https://www.zhihu.com/equation?tex=%5Calpha)

，即：

![[公式]](https://www.zhihu.com/equation?tex=CE%3D%5Cleft%5C%7B+%5Cbegin%7Baligned%7D+-%5Calpha+log%28p%29++%2C%5Cquad+if%5Cquad++y%3D1%5C%5C+-%281-%5Calpha%29log%281-p%29%2C%5Cquad+if%5Cquad++y%3D0+%5Cend%7Baligned%7D+%5Cright.)

但这并不解决全部问题。根据正、负、难、易，样本一共可以分为以下四类：

![img](https://pic1.zhimg.com/80/v2-21ca62ae70e63d1e99833e368fcac4cc_720w.jpg)

**尽管** ![[公式]](https://www.zhihu.com/equation?tex=%5Calpha) **平衡了正负样本，但对难易样本的不平衡没有任何帮助。**而实际上，目标检测中大量的候选目标都是像下图一样的易分样本。

![img](https://pic2.zhimg.com/80/v2-64c4966f51990c63027041178ee86659_720w.jpg)

这些样本的损失很低，但是由于数量极其不平衡，易分样本的数量相对来讲太多，最终主导了总的损失。而本文的作者认为，**易分样本（即，置信度高的样本）对模型的提升效果非常小，模型应该主要关注与那些难分样本**（**这个假设是有问题的，是GHM的主要改进对象**）。

这时候，Focal Loss该上场了。

一个简单的思想：**把高置信度(p)样本的损失再降低一些不就好了吗！**

![[公式]](https://www.zhihu.com/equation?tex=FL%3D%5Cleft%5C%7B+%5Cbegin%7Baligned%7D+-%281-p%29%5E%5Cgamma+log%28p%29++%2C%5Cquad+if%5Cquad++y%3D1%5C%5C+%5C%5C+-p%5E%5Cgamma+log%281-p%29%2C%5Cquad+if%5Cquad++y%3D0+%5Cend%7Baligned%7D+%5Cright.)

举个例， ![[公式]](https://www.zhihu.com/equation?tex=%5Cgamma) 取2时，如果 ![[公式]](https://www.zhihu.com/equation?tex=p%3D0.968) , ![[公式]](https://www.zhihu.com/equation?tex=%281-0.968%29%5E2+%5Capprox+0.001) ，损失衰减了1000倍！

Focal Loss的最终形式结合了上面的公式（2）. 这很好理解，公式(3)解决了难易样本的不平衡，公式(2)解决了正负样本的不平衡，将公式（2）与（3）结合使用，同时解决正负难易2个问题！

最终的Focal Loss形式如下：

![[公式]](https://www.zhihu.com/equation?tex=FL%3D%5Cleft%5C%7B+%5Cbegin%7Baligned%7D+-%5Calpha+%281-p%29%5E%5Cgamma+log%28p%29++%2C%5Cquad+if%5Cquad++y%3D1%5C%5C++-%281-%5Calpha%29+p%5E%5Cgamma+log%281-p%29%2C%5Cquad+if%5Cquad++y%3D0+%5Cend%7Baligned%7D+%5Cright.)

实验表明![[公式]](https://www.zhihu.com/equation?tex=%5Cgamma) 取2, ![[公式]](https://www.zhihu.com/equation?tex=%5Calpha) 取0.25的时候效果最佳。

![img](https://pic2.zhimg.com/80/v2-4c9b5f3265ed997958d986ee73d2ba65_720w.jpg)

这样以来，训练过程关注对象的排序为正难>负难>正易>负易。

这就是Focal Loss，简单明了但特别有用。

## Focal Loss的实现：

```
def py_sigmoid_focal_loss(pred,
                          target,
                          weight=None,
                          gamma=2.0,
                          alpha=0.25,
                          reduction='mean',
                          avg_factor=None):
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    focal_weight = (alpha * target + (1 - alpha) *
                    (1 - target)) * pt.pow(gamma)
    loss = F.binary_cross_entropy_with_logits(
        pred, target, reduction='none') * focal_weight
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss
```

这个代码很容易理解，

先定义一个pt：

![[公式]](https://www.zhihu.com/equation?tex=%5Cmathrm%7Bp%7D_%7Bt%7D%3D%5Cleft%5C%7B%5Cbegin%7Barray%7D%7Bc%7D%7B1-p%2C+%5Ctext+%7B+tatget+%7D%3D1%7D+%5C%5C+%7Bp%2C+%5Ctext+%7B+target+%7D%3D0%7D%5Cend%7Barray%7D%5Cright.)

然后计算：

```text
focal_weight = (alpha * target + (1 - alpha) *(1 - target)) * pt.pow(gamma)
```

也就是这个公式：

![[公式]](https://www.zhihu.com/equation?tex=FocalWeight%3D%5Cleft%5C%7B+%5Cbegin%7Baligned%7D+%5Calpha+%281-p%29%5E%5Cgamma+++%2C%5Cquad+if%5Cquad++target%3D1%5C%5C++%281-%5Calpha%29+p%5E%5Cgamma+%2C%5Cquad+if%5Cquad++target%3D0+%5Cend%7Baligned%7D+%5Cright.)

再把BCE损失*focal_weight就行了

![[公式]](https://www.zhihu.com/equation?tex=FL%3D%5Cleft%5C%7B+%5Cbegin%7Baligned%7D+-%5Calpha+%281-p%29%5E%5Cgamma+log%28p%29++%2C%5Cquad+if%5Cquad++target%3D1%5C%5C++-%281-%5Calpha%29+p%5E%5Cgamma+log%281-p%29%2C%5Cquad+if%5Cquad++target%3D0+%5Cend%7Baligned%7D+%5Cright.)

代码来自于mmdetection\mmdet\models\losses，这个python版的sigmoid_focal_loss实现就是让你拿去学习的，真正使用的是cuda编程版。真是个人性化的好框架

## 第二部分 GHM

那么，Focal Loss存在什么问题呢？

**首先**，让模型过多关注那些特别难分的样本肯定是存在问题的，样本中有**离群点（outliers）**，可能模型已经收敛了但是这些离群点还是会被判断错误，让模型去关注这样的样本，怎么可能是最好的呢？

![img](https://pic2.zhimg.com/80/v2-662e8862051dc3e9cc15d15ab8852bcd_720w.jpg)