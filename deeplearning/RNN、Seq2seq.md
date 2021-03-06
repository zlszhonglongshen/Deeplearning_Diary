# 完全图解RNN、RNN变体、Seq2seq、Attention机制

## 1、从单层网络谈起

在学习RNN之前，首先要了解一下最基本的单层网络，它的结构如图：

![img](https://pic1.zhimg.com/80/v2-da9ac1b5e3f91086fd06e6173fed1580_720w.jpg)

输入是x，经过变换Wx+b和激活函数f得到输出y。相信大家对这个已经非常熟悉了。

二、经典的RNN结构（N vs N)

在实际应用中，我们还会遇到很多序列形的数据：

![img](https://pic3.zhimg.com/80/v2-0f8f8a8313867459d33e902fed97bd16_720w.jpg)

如：

* 自然语言处理问题。x1可以看作是第一个单词，x2可以看做是第二个单词，依次类推。
* 语音处理。此时，x1、x2、x3....是每帧的声音信号。
* 时间序列问题。例如每天的股票价格等

序列形的数据就不太好用原始的神经网络了。为了建模序列问题，RNN引入了隐状态h（hidden state）的概念，h可以对序列形的数据提取特征，接着再转化为输出。先从h1的计算开始看：

![img](https://pic1.zhimg.com/80/v2-a5f8bc30bcc2d9eba7470810cb362850_720w.jpg)

图示中记号的含义是：

* 圆圈或方块表示的是向量
* 一个箭头就表示对该向量做一次变换。如上图中h0和x1分别有一个箭头连接，就表示对h0和x1各做了一次变化。

在很多论文中也会出现类似的记号，初学的时候很容易搞乱，但只要把握以上两点，就可以比较轻松的理解图示背后的含义。

h2的计算和h1类似。要注意的是，在计算时，每一步使用的参数U、W、b都是一样的，也就是说每一步骤的参数都是共享的。这就是RNN的重要特点，一定要牢记。

![img](https://pic3.zhimg.com/80/v2-74d7ac80ca83165092579932920d0ffe_720w.jpg)

依次计算剩下来的（使用相同的参数U、W、b）

![img](https://pic2.zhimg.com/80/v2-bc9759f8c642208a0f8514ccd0260b31_720w.jpg)

我们这里为了方便起见，只画出序列长度为4的情况，实际上，这个计算过程可以无限的持续下去。

我们目前的RNN还没有输出，得到输出值的方法就是直接连接h进行计算：

![img](https://pic1.zhimg.com/80/v2-9f3a921d0d5c1313afa58bd3ef53af48_720w.jpg)

正如之前所说，一个箭头就表示对应的向量做一次类似于f(Wx+b)的变换，这里的这个箭头就代表对h1进行一次变换，得到输出y1.

剩下的输出类似进行（使用和y1同样的参数V和c):

![img](https://pic2.zhimg.com/80/v2-629abbab0d5cc871db396f17e9c58631_1440w.jpg)

这就是典型的RNN结构，我们像搭积木一样把他搭建好了。它的输入是x1,x2,...,xn，输出为y1,y2...,yn，也就是说，输入和输出必须要是等长的。

由于这个限制的存在，经典RNN的使用范围比较小，但也有一些问题适合用经典的RNN结构建模，如：

* 计算视频中每一帧的分类标签，因为要对每一帧进行计算，因此输入和输出序列等长。
* 输入为字符，输出为下一个字符的概率。这就是著名的Char RNN（详细介绍请参考：[The Unreasonable Effectiveness of Recurrent Neural Networks](https://link.zhihu.com/?target=http%3A//karpathy.github.io/2015/05/21/rnn-effectiveness/)，Char RNN可以用来生成文章，诗歌，甚至是代码，非常有意思）。



## 2.N VS 1

有的时候，我们要处理的问题输入是一个序列，输出是一个单独的值，而不是序列，应该怎么建模呢？实际上，我们只在最后一个h上进行变换就行了：

![img](https://pic1.zhimg.com/80/v2-6caa75392fe47801e605d5e8f2d3a100_1440w.jpg)

这种结构通常用来处理序列分类问题。如输入一段文字判别它所属的类别，输入一个句子判断其情感倾向，输入一段视频并判断它的类别等等。

## 4、1VSN

输入不是序列而输出为序列的情况怎么处理？我们可以只在序列开始进行输入计算：

![img](https://pic3.zhimg.com/80/v2-87ebd6a82e32e81657682ffa0ba084ee_1440w.jpg)

还有一种结构是把输入信息X作为每个阶段的输入：

![img](https://pic3.zhimg.com/80/v2-fe054c488bb3a9fbcdfad299b2294266_1440w.jpg)

下图省略了一些X的圆圈，是一个等价表示：

![img](https://pic1.zhimg.com/80/v2-16e626b6e99fb1d23c8a54536f7d28dc_1440w.jpg)

这种1VSN的结构可以处理的问题有：

* 从图像生成文字，此时输入的X就是图像的特征，而输出的y序列就是一段句子
* 从类别生成语音或音乐等



## 5、N vs M

下面我们来介绍RNN最重要的一个变种：N vs M。这种结构又叫Encoder-Decoder模型，也可以称之为Seq2Seq模型。

原始的N vs N RNN要求序列等长，然而我们遇到的大部分问题序列都是不等长的，如机器翻译中，源语言和目标语言的句子往往并没有相同的长度。

为此，**Encoder-Decoder结构先将输入数据编码成一个上下向量C：**

![img](https://pic2.zhimg.com/80/v2-03aaa7754bb9992858a05bb9668631a9_1440w.jpg)

得到c有多种方式，最简单的方法就是把encoder的最后一个隐状态赋值给C，还可以对最后的隐状态做一个变换得到C，也可以对所有的隐状态做变换。

**拿到c之后，就用另一个RNN网络对其进行解码**，这部分RNN网络被称为Decoder。具体做法就是将c当做之前的初始状态h0输入到Decoder中：

![img](https://pic4.zhimg.com/80/v2-77e8a977fc3d43bec8b05633dc52ff9f_1440w.jpg)

还有一种做法是将c当做每一步的输入：

![img](https://pic4.zhimg.com/80/v2-e0fbb46d897400a384873fc100c442db_1440w.jpg)

由于这种Encoder-Decoder结构不限制输入和输出的序列长度，因此应用的范围非常广泛，比如：

- 机器翻译。Encoder-Decoder的最经典应用，事实上这一结构就是在机器翻译领域最先提出的
- 文本摘要。输入是一段文本序列，输出是这段文本序列的摘要序列。
- 阅读理解。将输入的文章和问题分别编码，再对其进行解码得到问题的答案。
- 语音识别。输入是语音信号序列，输出是文字序列。
- …………



https://zhuanlan.zhihu.com/p/28054589