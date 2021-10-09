机器学习入门-Stacking方法详解与具体案例

# stacking方法简述

关于stacking方法的定义以及原理讲解百度上很多，我这里主要挑选了两篇作为参考。

* [Stacking方法详解](https://www.cnblogs.com/Christina-Notebook/p/10063146.html)

* [详解stacking过程](https://blog.csdn.net/wstcjf/article/details/77989963)
* [集成学习中的 stacking 以及python实现](https://www.cnblogs.com/jiaxin359/p/8559029.html)

这里简单的总结一下：

Stacking是一种分层模型集成框架。以两层为例，第一层由多个基学习器组成，其输入为原始训练集，第二层的模型则是以第一层基学习器的输出作为特征加入训练集进行再训练，从而得到完整的stacking模型。stacking的方法在各大数据挖掘比赛上出尽风头，模型融合之后能够小幅度提高模型的预测准确率。

下图为一个简单的stacking流程图，其中LR模型代表线性回归模型。

![img](https://img2018.cnblogs.com/blog/1526211/201812/1526211-20181204113430234-89704009.png)

1、首先我们会得到两组数据：训练集和测试集。将训练集分成5份：train1,train2,train3,train4,train5。

2、选定基模型。这里假定我们选择了xgboost, lightgbm 和 randomforest 这三种作为基模型。比如xgboost模型部分：依次用train1,train2,train3,train4,train5作为验证集，其余4份作为训练集，进行5折交叉验证进行模型训练；再在测试集上进行预测。这样会得到在训练集上由xgboost模型训练出来的5份predictions，和在测试集上的1份预测值B1。将这五份纵向重叠合并起来得到A1。lightgbm和randomforest模型部分同理。

3、三个基模型训练完毕后，将三个模型在训练集上的预测值作为分别作为3个"特征"A1,A2,A3，使用LR模型进行训练，建立LR模型。

4、使用训练好的LR模型，在三个基模型之前在测试集上的预测值所构建的三个"特征"的值(B1,B2,B3)上，进行预测，得出最终的预测类别或概率。

![img](https://img-blog.csdnimg.cn/20190904201112532.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dzdGNqZg==,size_16,color_FFFFFF,t_70)