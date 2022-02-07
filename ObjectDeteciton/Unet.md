### unet

#### 概述

* 解决什么问题？

  医学图像的分割

* 使用的方法？

  * 集成FCN的思想，继续进行改进。但是相对于FCN，有几个改变的地方，unet是完全对称的，且对编码器进行了加卷积加深处理，FCN只是单纯的进行了上采样。
  * skip connection：两者都用了这样的结构，虽然在现在看来这样的做法比较常见，但是对于当时，这样的结构所带来的明显好处是有目共睹的，因为可以联合高层语义和底层的细粒度表层信息，就很好的符合了分割对这两方面信息的需求。

* 联合：在FCN中，Skip connection的联合是通过对应像素的求和，而unet则是对其channel的concat过程。

#### 网络结构

![unet网络结构图](https://img-blog.csdnimg.cn/20181127092719427.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2dpdGh1Yl8zNzk3MzYxNA==,size_16,color_FFFFFF,t_70)

#### 1、结构的解析

整体结构就是先编码（下采样），再编码（上采样），回归到跟原始图像一样大小的像素点的分类。

* 首先是输入图像的大小，这个是根据在高层的大小来进行反推的，最后取的一个比较合适的方便计算的输入大小。
* 下采样是通过max pool 2x2来进行1/2下采样的，下采样之间是两个conv卷积层，这里的卷积是使用valid卷积。所以在卷积过程中图像的大小是会减少的。这会造成一个问题，就是造成了在skip connection部分concat时候大小不一致，因为在上面有一个copy&crop操作，crop就是为了将大小进行裁剪的操作。
* 虽然上面有说到crop操作，但若是在卷积的时候使用的是same，就无需这个操作，至于这其中的影响，我觉得应该是不会造成太大的影响的，而且还会方便计算操作。
* 上采样，相对于FCN的转置卷积进行上采样，这里是一个up-conv 2x2







## 参考

https://mp.weixin.qq.com/s/9FsDEQFGUXWYJnzzQCCkcg
