# seq2seq模型以及注意力机制
传统的RNN及其变种GRU以及LSTM。它们只能处理输出序列为定长的情况。如果要处理输出序列为不定长情况的问题呢？
例如机器翻译，例如英文到法语的句子翻译，输入和输出均为不定长。前人提出了seq2seq模型，basic idea是设计一个encoder与decoder
，其中encoder将输入序列编码为一个包含输入序列所有信息的context vector c，decoder通过对 c 的解码获得输入序列的信息，从而得到输出序列。
encoder及decoder都通常为RNN循环神经网络。

## seq2seq模型

![seq2seq](https://pic4.zhimg.com/80/v2-e0fbb46d897400a384873fc100c442db_720w.jpg)

## 编码器Encoder

编码器的作用是把一个不定长的输入序列![[公式]](https://www.zhihu.com/equation?tex=x_1%2C+x_2%2C%5Cldots%2Cx_T)转化为一个定长context vector c。该context vector 编码了输入序列![[公式]](https://www.zhihu.com/equation?tex=x_1%2C+x_2%2C%5Cldots%2Cx_T)的序列。回忆一下RNN，假设该循环神经网络单元为f（可以为vanilla RNN，LSTM，GRU)，那么hidden state为

![[公式]](https://www.zhihu.com/equation?tex=h_t%3Df%28x_t%2C+h_%7Bt-1%7D%29)

编码器的context vector是所有时刻hidden state的函数，即

![[公式]](https://www.zhihu.com/equation?tex=c%3Dq%28h_1%2C%5Cldots%2Ch_T%29)

简单的，我们可以把最终时刻的hidden state ![[公式]](https://www.zhihu.com/equation?tex=h_T)作为context vector。当然我们也可以取各个时刻hidden states的平均，以及其他方法。

## 解码器Decoder

编码器最终输出一个context vector c，该context vector编码了输入序列![[公式]](https://www.zhihu.com/equation?tex=x_1%2C+x_2%2C%5Cldots%2Cx_T)的信息。

假设训练数据中的输出序列为![[公式]](https://www.zhihu.com/equation?tex=y_1%2C+y_2%2C%5Cldots%2Cy_T%27)，我们希望每个t时刻的输出即取决于之前的输出也取决于context vector，即估计![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbb%7BP%7D%28y_%7Bt%27%7D%7Cy_1%2C%5Cldots%2Cy_%7Bt%27-1%7D%2C+c%29)，从而得到输出序列的联合概率分布：

![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbb%7BP%7D%28y_1%2C%5Cldots%2Cy_%7BT%27%7D%29%3D%5Cprod_%7Bt%27%3D1%7D%5E%7BT%27%7D%5Cmathbb%7BP%7D%28y_%7Bt%27%7D%7Cy_1%2C%5Cldots%2Cy_%7Bt%27-1%7D%2Cc%29)

并定义该序列的损失函数loss function

![[公式]](https://www.zhihu.com/equation?tex=-%5Clog%5Cmathbb%7BP%7D%28y_1%2C%5Cldots%2Cy_%7BT%27%7D%29)

通过最小化损失函数来训练seq2seq模型。

那么如何估计![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbb%7BP%7D%28y_%7Bt%27%7D%7Cy_1%2C%5Cldots%2Cy_%7Bt%27-1%7D%2C+c%29)？

我们使用另一个训练神经网络作为编码器。解码器使用函数p来表示![[公式]](https://www.zhihu.com/equation?tex=t%27)时刻输出![[公式]](https://www.zhihu.com/equation?tex=y_%7Bt%27%7D)的概率。

![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbb%7BP%7D%28y_%7Bt%27%7D%7Cy_1%2C%5Cldots%2Cy_%7Bt%27-1%7D%2C+c%29%3Dp%28y_%7Bt%27-1%7D%2Cs_%7Bt%27%7D%2Cc%29)

为了区分编码器中的hidden state ![[公式]](https://www.zhihu.com/equation?tex=h_t)其中![[公式]](https://www.zhihu.com/equation?tex=s_%7Bt%27%7D)为![[公式]](https://www.zhihu.com/equation?tex=t%27)时刻解码器的hidden state。区别于编码器，解码器中的循环神经网络的输入除了前一个时刻的输出序列![[公式]](https://www.zhihu.com/equation?tex=y_%7Bt%27-1%7D)，和前一个时刻的hidden state![[公式]](https://www.zhihu.com/equation?tex=s_%7Bt%27-1%7D)以外，还包含了context vector![[公式]](https://www.zhihu.com/equation?tex=c)。即：

![[公式]](https://www.zhihu.com/equation?tex=s_%7Bt%27%7D%3Dg%28y_%7Bt%27-1%7D%2Cs_%7Bt%27-1%7D%2Cc%29)

其中函数g为解码器的循环神经网络单元。

## 参考

https://zhuanlan.zhihu.com/p/36440334

