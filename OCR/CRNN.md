* [一文读懂CRNN+CTC文字识别](https://zhuanlan.zhihu.com/p/43534801)
* https://www.cnblogs.com/skyfsm/p/10335717.html

### 一文读懂CRNN+CTC文字识别



#### CRNN基本网络结构

![img](https://pic3.zhimg.com/80/v2-7ed5c65fe79dce49f006a9171cc1a80e_720w.jpg)

整个CRNN网络可以分为三个部分：

假设输入图像大小为（32,100,3），注意提及图像都是（H,W,C)

* Convlutional Layers

  这里的卷积层就是一个普通的CNN网络，用于提取输入图像的Convolutional feature maps，即将大小为（32,100,3）的图像转换为（1,25,512）大小的卷积特征矩阵，网络细节请参考文本给出的实现代码。

* Recurrent Layers

  这里的循环网络层是一个深层双向LSTM网络，在卷积特征的基础上继续提取文字序列特征。对RNN不了解的读者，建议参考：

  [完全解析RNN, Seq2Seq, Attention注意力机制zhuanlan.zhihu.com![图标](https://pic4.zhimg.com/v2-2d34ffd69bc1d4d1c9231d0562db2fcf_180x120.jpg)](https://zhuanlan.zhihu.com/p/51383402)

  所谓深层RNN网络，是指超过两层的RNN网络。对于单层双向RNN网络，结构如下：

  ![img](https://pic4.zhimg.com/80/v2-9f5125e0c99924d2febf25bafd019d6f_720w.jpg)图5 单层双向RNN网络

  而对于深层双向RNN网络，主要有2种不同的实现：

  ```python
  tf.nn.bidirectional_dynamic_rnn
  ```

  ![img](https://pic3.zhimg.com/80/v2-c0132f0b748eb031c696dae3019a2d82_720w.jpg)图6 深层双向RNN网络

  ```python
  tf.contrib.rnn.stack_bidirectional_dynamic_rnn
  ```

  ![img](https://pic2.zhimg.com/80/v2-00861a152263cff8b94525d8b8945ee9_720w.jpg)图7 stack形深层双向RNN网络

  在CRNN中显然使用了第二种stack形深层双向结构。

  由于CNN输出的Feature map是（1,52,512）大小，所以对于RNN最大时间长度T=25（即有25个时间输入，每个输入Xt列向量有D=512）

* Transcription Layers

  将RNN输出做softmax后，为字符输出。

  #### 关于代码中输入图片大小的解释

  在本文给出的实现中，为了将特诊输入到Recurrent Layers，做如下处理：

  * 首先会将图像在固定长宽比的情况下缩放到 ![[公式]](https://www.zhihu.com/equation?tex=32%5Ctimes+W%5Ctimes3) 大小（ ![[公式]](https://www.zhihu.com/equation?tex=W) 代表任意宽度）
  * 然后经过CNN后变为 ![[公式]](https://www.zhihu.com/equation?tex=1%5Ctimes+%28W%2F4%29%5Ctimes512)
  * 针对LSTM设置 ![[公式]](https://www.zhihu.com/equation?tex=T%3D%28W%2F4%29) ，即可将特征输入LSTM。

  所以在处理输入图像的时候，建议在保持长宽比的情况下将高缩放到 ![[公式]](https://www.zhihu.com/equation?tex=32)，这样能够尽量不破坏图像中的文本细节（当然也可以将输入图像缩放到固定宽度，但是这样由于破坏文本的形状，肯定会造成性能下降）。

  