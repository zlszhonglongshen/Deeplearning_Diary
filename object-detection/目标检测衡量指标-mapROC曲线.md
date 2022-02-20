## 常见的模型算法评估指标

|        | 预测值   |          |
| ------ | -------- | -------- |
| 真实值 | positive | Negative |
| True   | TP       | TN       |
| False  | FP       | FN       |

* recall = TP/(TP+FN),召回率

* F1 = 2x(precision*recall)/(precision+recall),F1-score

* TPR（真阳率）,指的是在所有实际为阳性的样本中，被正确的判断为阳性的比率，同召回率

* TPR（真阳率）=TP/(TP+FN)

* FPR(假阳率)，指的是在所有实际为阴性的样本中，被错误的判断为阳性的比率。其公式如下：

​                                                                           FPR = FP/(FP+TN) 

**ROC曲线**

其以FPR假阳率为X轴坐标，以TPR真阳率为Y轴坐标，曲线越靠近左上角则说明模型算法性能越好，左上角（0,1）为最理想的情况说明模型性能非常完美，而其对角线对应于“随机猜测”模型的性能。

\* IOU,交并比，指的是ground truth bbox和predict bbox的交集占两者并集面积的一个比率。IOU值越大说明预测检测框的模型算法性能越好。通常目标检测任务里面讲IOU>=0.7的区域设定为正例，而将IOU<=0.3的区域设定为负类样本，其余的会丢弃掉

\* AP，AP为平均精度，指的是所有图片内的具体某一类的PR曲线下的面积，计算方式，首先设定一组recall阈值，然后对每个recall阈值从小到大取值，同时计算当取大于该recall阈值时top-n所对应的最大precision，这样，就得到11个precison，AP即为这话11个precision的平均值。

\* mAP，mAP为均值平均精度，指的是所有图片内的所有类别的AP平均值。



### 参考链接

[详解目标检测中的mAP](https://zhuanlan.zhihu.com/p/56961620)